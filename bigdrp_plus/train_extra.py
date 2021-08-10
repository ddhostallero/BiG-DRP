from utils.data_initializer import initialize
from utils.utils import mkdir, reindex_tuples, create_fold_mask, reset_seed
from utils.tuple_dataset import TupleMapDataset
import torch
from torch.utils.data import DataLoader, TensorDataset
from bigdrp.model import DRPPlus
import numpy as np
import pandas as pd

def predict_matrix(model, data_loader, drug_encoding, device):
    """
    returns a prediction matrix of (N, n_drugs)
    """

    model.eval()

    preds = []
    drug_encoding = drug_encoding

    with torch.no_grad():
        for (x,) in data_loader:
            x = x.to(device)
            pred = model.predict_response_matrix(x, drug_encoding)
            preds.append(pred)

    preds = torch.cat(preds, axis=0).cpu().detach().numpy()
    return preds


def train_extra_and_test(FLAGS, cell_lines, labels, label_matrix, normalizer):
    """
    Train for 1 epoch with fixed embeddings (on the training set) then test on the test set
    """

    hyperparams = {
        'learning_rate': 1e-5,
        'num_epoch': 1,
        'batch_size': 128,
        'common_dim': 512,
        'expr_enc': 1024,
        'conv1': 512,
        'conv2': 512,
        'mid': 512,
        'drop': [0.2, 0.5]}
    
    label_mask = create_fold_mask(labels, label_matrix)
    label_matrix = label_matrix.replace(np.nan, 0)

    n_genes = cell_lines.shape[1]
    metrics = np.zeros((5, 4))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(5):
        reset_seed(FLAGS.seed)
        test_fold = i 
        model_path = FLAGS.outroot + "results/" + FLAGS.weight_folder + '/model_weights_fold_%d'%i
        encoding_path = FLAGS.outroot + "results/" + FLAGS.weight_folder + '/encoding_fold_%d.csv'%i

        print("Loading encoding from: %s..."%encoding_path)
        drug_encoding = pd.read_csv(encoding_path, index_col=0)
        dr_idx = drug_encoding.index

        train_folds = [x for x in range(5) if (x != test_fold)]
        train_tuples = labels.loc[labels['fold'].isin(train_folds)]
        train_samples = list(train_tuples['cell_line'].unique())
        train_x = cell_lines.loc[train_samples].values
        train_y = label_matrix.loc[train_samples].values
        train_mask = (label_mask.loc[train_samples].isin(train_folds))*1

        test_tuples = labels.loc[labels['fold'] == test_fold]
        test_samples = list(test_tuples['cell_line'].unique())
        test_x = cell_lines.loc[test_samples].values
        test_mask = (label_mask.loc[test_samples]==test_fold)*1

        train_x, test_x = normalizer(train_x, test_x)

        train_tuples = train_tuples[['drug', 'cell_line', 'response']]
        train_tuples = reindex_tuples(train_tuples, dr_idx, train_samples)
        train_data = TupleMapDataset( 
            train_tuples,
            drug_encoding,
            torch.FloatTensor(train_x),
            torch.FloatTensor(train_y))
        train_data = DataLoader(train_data, batch_size=128)

        print("Loading weights from: %s..."%model_path)
        model = DRPPlus(cell_lines.shape[1], hyperparams)
        model.load_state_dict(torch.load(model_path), strict=False)
        model = model.to(device)

        # === train for 1 epoch ===
        params = model.parameters()
        optimizer = torch.optim.Adam(params, lr=1e-5)
        model.train()
        for (x, d, y) in train_data:
            x, d, y = x.to(device), d.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x, d)
            loss = ((pred - y)**2).mean()
            loss.backward()
            optimizer.step()
            print("train MSE: %.4f"%loss.item())


        # === test on test set ===
        drug_encoding = torch.Tensor(drug_encoding.values).to(device)
        test_data = TensorDataset(torch.FloatTensor(test_x))
        test_data = DataLoader(test_data, batch_size=hyperparams['batch_size'], shuffle=False)

        prediction_matrix = predict_matrix(model, test_data, drug_encoding, device)
        prediction_matrix = pd.DataFrame(prediction_matrix, index=test_samples, columns=dr_idx)
        test_mask = test_mask.replace(0, np.nan)
        prediction_matrix = prediction_matrix*test_mask
        prediction_matrix.to_csv(FLAGS.outroot + "results/" + FLAGS.folder + '/val_prediction_fold_%d.csv'%i)

        directory = FLAGS.outroot + "/results/" + FLAGS.folder
        torch.save(model.state_dict(), directory+'/model_weights_fold_%d'%test_fold)


def main(FLAGS):
    drug_feats, cell_lines, labels, label_matrix, normalizer = initialize(FLAGS)
    train_extra_and_test(FLAGS, cell_lines, labels, label_matrix, normalizer)
