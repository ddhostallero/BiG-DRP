import torch
from sklearn.metrics import r2_score
from bigdrp.model import BiGDRP
import time
import numpy as np
from scipy.stats import pearsonr, spearmanr
from collections import deque
import json
from dgl.dataloading import MultiLayerFullNeighborSampler
import torch.nn.functional as F

class Trainer:
    def __init__(self, n_genes, cell_feats, drug_feats, network, hyp, test=False, load_model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cell_feats = cell_feats
        self.drug_feats = drug_feats
        self.network = network

        if load_model_path is not None:
            self.model = ModelHead(n_genes, hyp)
            self.model.load_state_dict(torch.load(load_model_path), strict=False)
            self.model = self.model.to(self.device)

        if not test:
            self.model = BiGDRP(n_genes, self.cell_feats.shape[1], drug_feats.shape[1], network.etypes, hyp).to(self.device)
            
            self.mse_loss = torch.nn.MSELoss()
            params = self.model.parameters()
            self.optimizer = torch.optim.Adam(params, lr=hyp['learning_rate'])

            graph_sampler = MultiLayerFullNeighborSampler(2)
            print(self.cell_feats.shape)
            self.network.ndata['features'] = {'drug': self.drug_feats, 'cell_line': self.cell_feats}
            _,_, blocks = graph_sampler.sample_blocks(self.network, {'drug': range(len(drug_feats))})
            self.blocks = [b.to(self.device) for b in blocks]
           
            #make sure they are aligned correctly
            self.cell_feats = self.blocks[0].ndata['features']['cell_line'].to(self.device)
            self.drug_feats = self.blocks[0].ndata['features']['drug'].to(self.device)


    def train_step(self, train_loader, device):
        # trains on tuples
        self.model.train()
        for (x, d1, y, w) in train_loader:
            x, d1, y = x.to(device), d1.to(device), y.to(device)
            w = w.to(device)

            self.optimizer.zero_grad()
            pred = self.model(self.blocks, self.drug_feats, self.cell_feats, x, d1)
            loss = self.mse_loss(pred, y)

            loss.backward()
            self.optimizer.step()

        r2 = r2_score(y.cpu().detach().numpy(), pred.cpu().detach().numpy())
        return loss.item(), r2

    def validation_step_cellwise(self, val_loader, device):
        """
        Creates a matrix of predictions with shape (n_cell_lines, n_drugs) and calculates the metrics
        """

        self.model.eval()

        val_loss = 0
        preds = []
        ys = []
        with torch.no_grad():
            for (x, y, mask) in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                pred = self.model.predict_all(self.blocks, self.drug_feats, self.cell_feats, x)
                val_loss += (((pred - y)*mask)**2).sum()

                mask = mask.cpu().detach().numpy().nonzero()
                ys.append(y.cpu().detach().numpy()[mask])
                preds.append(pred.cpu().detach().numpy()[mask])

        preds = np.concatenate(preds, axis=0)
        ys = np.concatenate(ys, axis=0)
        r2 = r2_score(ys, preds)
        pearson = pearsonr(ys, preds)[0]
        spearman = spearmanr(ys, preds)[0]

        return val_loss.item()/len(ys), r2, pearson, spearman

    def get_drug_encoding(self):
        """
        returns the tensor of drug encodings (by GraphConv])
        """
        self.model.eval()
        with torch.no_grad():
            drug_encoding = self.model.get_drug_encoding(self.blocks, self.drug_feats, self.cell_feats)
        return drug_encoding

    def predict_matrix(self, data_loader, drug_encoding=None):
        """
        returns a prediction matrix of (N, n_drugs)
        """

        self.model.eval()

        preds = []
        if drug_encoding is None:
            drug_encoding = self.get_drug_encoding() # get the encoding first so that we don't have top run the conv every time
        else:
            drug_encoding = drug_encoding.to(self.device)

        with torch.no_grad():
            for (x,) in data_loader:
                x = x.to(self.device)
                pred = self.model.predict_response_matrix(x, drug_encoding)
                preds.append(pred)

        preds = torch.cat(preds, axis=0).cpu().detach().numpy()
        return preds


    def fit(self, num_epoch, train_loader, val_loader, tuning=False, maxout=False):
        start_time = time.time()

        ret_matrix = np.zeros((num_epoch, 6))
        loss_deque = deque([], maxlen=5)

        best_loss = np.inf
        best_loss_avg5 = np.inf
        best_loss_epoch = 0
        best_avg5_loss_epoch = 0

        count = 0

        for epoch in range(num_epoch):
            train_metrics = self.train_step(train_loader, self.device)
            val_metrics = self.validation_step_cellwise(val_loader, self.device)

            ret_matrix[epoch,:4] = val_metrics
            ret_matrix[epoch,4:] = train_metrics

            if best_loss > val_metrics[0]:
                best_loss = val_metrics[0]
                best_loss_epoch = epoch+1

            loss_deque.append(val_metrics[0])
            loss_avg5 = sum(loss_deque)/len(loss_deque)
            
            if best_loss_avg5 > loss_avg5:
                best_loss_avg5 = loss_avg5
                best_avg5_loss_epoch = epoch+1
                count = 0
            else:
                count += 1

            if count == 10 and not maxout:
                ret_matrix = ret_matrix[:epoch+1]
                break

            elapsed_time = time.time() - start_time
            start_time = time.time()
            print("%d\tval-mse:%.4f\tbatch-mse:%.4f\tval-r2:%.4f\tbatch-r2:%.4f\tval-spearman:%.4f\t%ds"%(
                epoch+1, val_metrics[0], train_metrics[0],val_metrics[1], train_metrics[1], val_metrics[3], int(elapsed_time)))

#        if not tuning:
        metric_names = ['test MSE', 'test R^2', 'test pearsonr', 'test spearmanr', 'train MSE', 'train R^2']
        return ret_matrix, metric_names

    def save_model(self, directory, fold_id, hyp):
        torch.save(self.model.state_dict(), directory+'/model_weights_fold_%d'%fold_id)

        x = json.dumps(hyp)
        f = open(directory+"/model_config_fold_%d.txt"%fold_id,"w")
        f.write(x)
        f.close()
