import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import HeteroGraphConv, GraphConv

class DRPPlus(nn.Module):
    def __init__(self, n_genes, hyp):
        super(DRPPlus, self).__init__()

        self.expr_l1 = nn.Linear(n_genes, hyp['expr_enc'])
        self.mid = nn.Linear(hyp['expr_enc'] + hyp['conv2'], hyp['mid'])
        self.out = nn.Linear(hyp['mid'], 1)
        
        if hyp['drop'] == 0:
            drop=[0,0]
        else:
            drop=[0.2,0.5]

        self.in_drop = nn.Dropout(drop[0])
        self.mid_drop = nn.Dropout(drop[1])
        self.alpha = 0.5

    def forward(self, cell_features, drug_enc):
        expr_enc = F.leaky_relu(self.expr_l1(cell_features))
        
        x = torch.cat([expr_enc,drug_enc],-1) # (batch, expr_enc_size+drugs_enc_size)
        x = self.in_drop(x)
        x = F.leaky_relu(self.mid(x)) # (batch, n_drugs, 1)
        x = self.mid_drop(x)
        out = self.out(x) # (batch, n_drugs, 1)
 
        return out

    def predict_response_matrix(self, cell_features, drug_enc):
        expr_enc = F.leaky_relu(self.expr_l1(cell_features))
        expr_enc = expr_enc.unsqueeze(1) # (batch, 1, expr_enc_size)
        drug_enc = drug_enc.unsqueeze(0) # (1, n_drugs, drug_enc_size)
        
        expr_enc = expr_enc.repeat(1,drug_enc.shape[1],1) # (batch, n_drugs, expr_enc_size)
        drug_enc = drug_enc.repeat(expr_enc.shape[0],1,1) # (batch, n_drugs, drug_enc_size)
        
        x = torch.cat([expr_enc,drug_enc],-1) # (batch, n_drugs, expr_enc_size+drugs_enc_size)
        x = self.in_drop(x)
        x = F.leaky_relu(self.mid(x)) # (batch, n_drugs, 1)
        x = self.mid_drop(x)
        out = self.out(x) # (batch, n_drugs, 1)
        out = out.view(-1, drug_enc.shape[1]) # (batch, n_drugs)
        return out
        


class BiGDRP(nn.Module):
    def __init__(self, n_genes, n_cl_feats, n_drug_feats, rel_names, hyp):
        super(BiGDRP, self).__init__()

        self.conv1 = HeteroGraphConv(
            {rel: dgl.nn.GraphConv(in_feats=hyp['common_dim'], out_feats=hyp['common_dim']) for rel in rel_names})
        self.conv2 = HeteroGraphConv(
            {rel: dgl.nn.GraphConv(in_feats=hyp['common_dim'], out_feats=hyp['common_dim']) for rel in rel_names})

        self.drug_l1 = nn.Linear(n_drug_feats, hyp['common_dim'])
        self.cell_l1 = nn.Linear(n_cl_feats, hyp['common_dim'])
        self.expr_l1 = nn.Linear(n_genes, hyp['expr_enc'])
        self.mid = nn.Linear(hyp['expr_enc'] + hyp['common_dim'], hyp['mid'])
        self.out = nn.Linear(hyp['mid'], 1)

        if hyp['drop'] == 0:
            drop=[0,0]
        else:
            drop=[0.2,0.5]

        self.in_drop = nn.Dropout(drop[0])
        self.mid_drop = nn.Dropout(drop[1])
        self.alpha = 0.5
        

    def forward(self, blocks, drug_features, cell_features_in_network, cell_features, drug_index):
        cell_enc = F.leaky_relu(self.cell_l1(cell_features_in_network))
        drug_enc = F.leaky_relu(self.drug_l1(drug_features))
        node_features = {'drug': drug_enc, 'cell_line': cell_enc}

        h1 = self.conv1(blocks[0], node_features)
        h1 = {k: F.leaky_relu(v + self.alpha*node_features[k]) for k, v in h1.items()}

        h2 = self.conv2(blocks[1], h1)
        h2['drug'] = F.leaky_relu(h2['drug'] + self.alpha*h1['drug'])
        # h2 = {k: F.leaky_relu(v + self.alpha*h1[k]) for k, v in h2.items()}
        
        expr_enc = F.leaky_relu(self.expr_l1(cell_features))        
        drug_enc = h2['drug'][drug_index]

        x = torch.cat([expr_enc,drug_enc],-1) # (batch, expr_enc_size+drugs_enc_size)
        x = self.in_drop(x)
        x = F.leaky_relu(self.mid(x)) 
        x = self.mid_drop(x)
        out = self.out(x)
        return out

    def predict_all(self, blocks, drug_features, cell_features_in_network, cell_features):

        cell_enc = F.leaky_relu(self.cell_l1(cell_features_in_network))
        drug_enc = F.leaky_relu(self.drug_l1(drug_features))
        node_features = {'drug': drug_enc, 'cell_line': cell_enc}
        
        h1 = self.conv1(blocks[0], node_features)
        h1 = {k: F.leaky_relu(v + self.alpha*node_features[k]) for k, v in h1.items()}
        
        h2 = self.conv2(blocks[1], h1)
        h2['drug'] = F.leaky_relu(h2['drug'] + self.alpha*h1['drug'])
        # h2 = {k: F.leaky_relu(v + self.alpha*h1[k]) for k, v in h2.items()}
        
        expr_enc = F.leaky_relu(self.expr_l1(cell_features))
        expr_enc = expr_enc.unsqueeze(1) # (batch, 1, expr_enc_size)
        drug_enc = h2['drug'].unsqueeze(0) # (1, n_drugs, drug_enc_size)
        
        expr_enc = expr_enc.repeat(1,drug_enc.shape[1],1) # (batch, n_drugs, expr_enc_size)
        drug_enc = drug_enc.repeat(expr_enc.shape[0],1,1) # (batch, n_drugs, drug_enc_size)
        
        x = torch.cat([expr_enc,drug_enc],-1) # (batch, n_drugs, expr_enc_size+drugs_enc_size)
        x = self.in_drop(x)
        x = F.leaky_relu(self.mid(x)) # (batch, n_drugs, 1)
        x = self.mid_drop(x)
        out = self.out(x) # (batch, n_drugs, 1)
        out = out.view(-1, drug_enc.shape[1])
        return out

    def get_drug_encoding(self, blocks, drug_features, cell_features_in_network):

        cell_enc = F.leaky_relu(self.cell_l1(cell_features_in_network))
        drug_enc = F.leaky_relu(self.drug_l1(drug_features))
        node_features = {'drug': drug_enc, 'cell_line': cell_enc}
        
        h1 = self.conv1(blocks[0], node_features)
        h1 = {k: F.leaky_relu(v + self.alpha*node_features[k]) for k, v in h1.items()}
        
        h2 = self.conv2(blocks[1], h1)
        h2['drug'] = F.leaky_relu(h2['drug'] + self.alpha*h1['drug'])
        # h2 = {k: F.leaky_relu(v + self.alpha*h1[k]) for k, v in h2.items()}
        
        drug_enc = h2['drug']
        return drug_enc

    def predict_response_matrix(self, cell_features, drug_enc):

        expr_enc = F.leaky_relu(self.expr_l1(cell_features))
        expr_enc = expr_enc.unsqueeze(1) # (batch, 1, expr_enc_size)
        drug_enc = drug_enc.unsqueeze(0) # (1, n_drugs, drug_enc_size)
        
        expr_enc = expr_enc.repeat(1,drug_enc.shape[1],1) # (batch, n_drugs, expr_enc_size)
        drug_enc = drug_enc.repeat(expr_enc.shape[0],1,1) # (batch, n_drugs, drug_enc_size)
        
        x = torch.cat([expr_enc,drug_enc],-1) # (batch, n_drugs, expr_enc_size+drugs_enc_size)
        x = self.in_drop(x)
        x = F.leaky_relu(self.mid(x)) # (batch, n_drugs, 1)
        x = self.mid_drop(x)
        out = self.out(x) # (batch, n_drugs, 1)
        out = out.view(-1, drug_enc.shape[1]) # (batch, n_drugs)
        return out

    
