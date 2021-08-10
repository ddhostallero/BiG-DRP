from utils.tuple_dataset import TupleMatrixDataset
from utils.utils import mkdir, reindex_tuples, moving_average, reset_seed, create_fold_mask
from utils.network_gen import create_network
from utils.data_initializer import initialize

import torch
from torch.utils.data import DataLoader, TensorDataset
from bigdrp.trainer import Trainer
import numpy as np
import pandas as pd

def fold_validation(hyperparams, seed, network, train_data, val_data, cell_lines, 
    drug_feats, tuning, epoch, final=False):

    r"""

    Description
    -----------
    Initializes a `Trainer` object then trains using given data and hyperparameters

    Parameters
    ----------
    hyperparams : dict
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    seed : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    network : pandas.DataFrame
        How to apply the normalizer. If is `'right'`, divide the aggregated messages

        where the :math:`c_{ji}` in the paper is applied.
    train_data : torch.DataSet
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix.
    val_data : torch.DataSet
        Contains
    cell_lines : torch.Tensor
        Tensor that contains the gene expressions. Must be in shape (n_cell_lines, n_genes)
    drug_feats : torch.Tensor
        Tensor that contains the drug features. Must be in shape (n_drugs, n_features)
    tuning : bool
        Set to true if we are using RayTune
    epoch : int
        Used as a maximum number of training epochs if ``final`` is False. 
        Otherwise, used as the fixed number of  trainingepochs.
    final : bool, optional
        If True, uses the ``epoch`` parameter as a fixed epoch instead of maximum epoch. 
        Otherwise, uses an early stopping criterion. Default: ``False``.
    """

    reset_seed(seed)
    train_loader = DataLoader(train_data, batch_size=hyperparams['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=hyperparams['batch_size'], shuffle=False)

    n_genes = cell_lines.shape[1]
    n_drug_feats = drug_feats.shape[1]

    trainer = Trainer(n_genes, cell_lines, drug_feats, network, hyperparams)
    val_error, metric_names = trainer.fit(
            num_epoch=epoch, 
            train_loader=train_loader, 
            val_loader=val_loader,
            tuning=tuning)
    if final:
        return val_error, trainer, metric_names
    else:
        return val_error, None, metric_names

def create_dataset(tuples, train_x, val_x, 
    train_y, val_y, train_mask, val_mask, drug_feats, percentile):

    r"""

    Description
    -----------
    Creates the training and validation/testing dataset and the training bipartite graph

    Parameters
    ----------
    tuples : pandas.DataFrame
        DataFrame containing the index of the drug index and the cell line index of the training pairs
    train_x : matrix
        Gene expressions of the training cell lines. 
    val_x : matrix
        Gene expressions of the validation cell lines
    train_y : matrix
        Training labels. Must be in shape (n_cell_lines, n_drugs)
    val_y : matrix
        Validation labels.Must be in shape (n_cell_lines, n_drugs)
    val_mask : matrix
        Matrix indicating that the (i,j)th cell in the ``val_y`` is part of the validation set
    drug_feats : matrix
        Drug features
    percentile :
        Percentile used in creating thresholds for the bipartite graph
    """

    network = create_network(tuples, percentile)

    train_data = TupleMatrixDataset( 
        tuples,
        torch.FloatTensor(train_x),
        torch.FloatTensor(train_y))

    val_data = TensorDataset(
        torch.FloatTensor(val_x),
        torch.FloatTensor(val_y),
        torch.FloatTensor(val_mask))

    cell_lines = torch.FloatTensor(train_x)
    drug_feats = torch.FloatTensor(drug_feats.values)

    return network, train_data, val_data, cell_lines, drug_feats

def nested_cross_validation(FLAGS, drug_feats, cell_lines, labels, label_matrix, normalizer):
    reset_seed(FLAGS.seed)
    hyperparams = {
        'learning_rate': 1e-4,
        'num_epoch': 50,
        'batch_size': 128,
        'common_dim': 512,
        'expr_enc': 1024,
        'conv1': 512,
        'conv2': 512,
        'mid': 512,
        'drop': 1}

    label_mask = create_fold_mask(labels, label_matrix)
    label_matrix = label_matrix.replace(np.nan, 0)

    final_metrics = None
    drug_list = list(drug_feats.index)

    for i in range(5):
        print('==%d=='%i)
        test_fold = i 
        val_fold = (i+1)%5
        train_folds = [x for x in range(5) if (x != test_fold) and (x != val_fold)]

        hp = hyperparams.copy()

        # === find number of epochs ===

        train_tuples = labels.loc[labels['fold'].isin(train_folds)]
        train_samples = list(train_tuples['cell_line'].unique())
        train_x = cell_lines.loc[train_samples].values
        train_y = label_matrix.loc[train_samples].values
        train_mask = (label_mask.loc[train_samples].isin(train_folds))*1

        val_tuples = labels.loc[labels['fold'] == val_fold]
        val_samples = list(val_tuples['cell_line'].unique())
        val_x = cell_lines.loc[val_samples].values
        val_y = label_matrix.loc[val_samples].values
        val_mask = ((label_mask.loc[val_samples]==val_fold)*1).values

        train_tuples = train_tuples[['drug', 'cell_line', 'response']]
        train_tuples = reindex_tuples(train_tuples, drug_list, train_samples) # all drugs exist in all folds

        train_x, val_x = normalizer(train_x, val_x)

        network, train_data, val_data, \
        cl_tensor, df_tensor = create_dataset(
            train_tuples, 
            train_x, val_x, 
            train_y, val_y, 
            train_mask, val_mask, drug_feats, FLAGS.network_perc)

        val_error,_,_ = fold_validation(hp, FLAGS.seed, network, train_data, 
            val_data, cl_tensor, df_tensor, tuning=False, 
            epoch=hp['num_epoch'], final=False)

        average_over = 3
        mov_av = moving_average(val_error[:,0], average_over)
        smooth_val_loss = np.pad(mov_av, average_over//2, mode='edge')
        epoch = np.argmin(smooth_val_loss)
        hp['num_epoch'] = int(max(epoch, 2)) 

        # === actual test fold ===

        train_folds = train_folds + [val_fold]
        train_tuples = labels.loc[labels['fold'].isin(train_folds)]
        train_samples = list(train_tuples['cell_line'].unique())
        train_x = cell_lines.loc[train_samples].values
        train_y = label_matrix.loc[train_samples].values
        train_mask = (label_mask.loc[train_samples].isin(train_folds))*1

        test_tuples = labels.loc[labels['fold'] == test_fold]
        test_samples = list(test_tuples['cell_line'].unique())
        test_x = cell_lines.loc[test_samples].values
        test_y = label_matrix.loc[test_samples].values
        test_mask = (label_mask.loc[test_samples]==test_fold)*1

        train_tuples = train_tuples[['drug', 'cell_line', 'response']]
        train_tuples = reindex_tuples(train_tuples, drug_list, train_samples) # all drugs exist in all folds

        train_x, test_x = normalizer(train_x, test_x)
        network, train_data, test_data, \
        cl_tensor, df_tensor = create_dataset(
            train_tuples, 
            train_x, test_x, 
            train_y, test_y, 
            train_mask, test_mask.values, drug_feats, FLAGS.network_perc)

        test_error, trainer, metric_names = fold_validation(hp, FLAGS.seed, network, train_data, 
            test_data, cl_tensor, df_tensor, tuning=False, 
            epoch=hp['num_epoch'], final=True) # set final so that the trainer uses all epochs

        if i == 0:
            final_metrics = np.zeros((5, test_error.shape[1]))

        final_metrics[i] = test_error[-1]
        test_metrics = pd.DataFrame(test_error, columns=metric_names)
        test_metrics.to_csv(FLAGS.outroot + "/results/" + FLAGS.folder + '/fold_%d.csv'%i, index=False)

        drug_enc = trainer.get_drug_encoding().cpu().detach().numpy()
        pd.DataFrame(drug_enc, index=drug_list).to_csv(FLAGS.outroot + "/results/" + FLAGS.folder + '/encoding_fold_%d.csv'%i)

        trainer.save_model(FLAGS.outroot + "/results/" + FLAGS.folder, i, hp)

        # save predictions
        test_data = TensorDataset(torch.FloatTensor(test_x))
        test_data = DataLoader(test_data, batch_size=hyperparams['batch_size'], shuffle=False)

        prediction_matrix = trainer.predict_matrix(test_data, drug_encoding=torch.Tensor(drug_enc))
        prediction_matrix = pd.DataFrame(prediction_matrix, index=test_samples, columns=drug_list)

        # remove predictions for non-test data
        test_mask = test_mask.replace(0, np.nan)
        prediction_matrix = prediction_matrix*test_mask

        prediction_matrix.to_csv(FLAGS.outroot + "/results/" + FLAGS.folder + '/val_prediction_fold_%d.csv'%i)
    
    return final_metrics

def main(FLAGS):

    drug_feats, cell_lines, labels, label_matrix, normalizer = initialize(FLAGS)
    test_metrics = nested_cross_validation(FLAGS, drug_feats, cell_lines, labels, label_matrix, normalizer)
    test_metrics = test_metrics.mean(axis=0)

    print("Overall Performance")
    print("MSE: %f"%test_metrics[0])
    print("RMSE: %f"%np.sqrt(test_metrics[0]))
    print("R2: %f"%test_metrics[1])
    print("Pearson: %f"%test_metrics[2])
    print("Spearman: %f"%test_metrics[3])