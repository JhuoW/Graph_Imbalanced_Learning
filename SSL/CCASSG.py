import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, _, x):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)

        x = self.act_fn(x)
        x = self.layer2(x)

        return x


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers) -> None:
        super(GCN, self).__init__()
        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hid_dim))
        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GCNConv(hid_dim, hid_dim))
            self.convs.append(GCNConv(hid_dim, out_dim))

    def forward(self, x, edge_index):
        h = x
        for i in range(self.n_layers - 1):
            h = F.relu(self.convs[i](h, edge_index))
        h = self.convs[-1](h, edge_index)
        return h

class CCA_SSG(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, use_mlp = False) -> None:
        super(CCA_SSG, self).__init__()
        if not use_mlp:
            self.backbone = GCN(in_dim, hid_dim, out_dim, n_layers)
        else:
            self.backbone = MLP(in_dim, hid_dim, out_dim)
    
    def get_embedding(self, x, edge_index):
        out = self.backbone(x, edge_index)
        return out.detach()
    
    def loss(self, z1: torch.Tensor, z2: torch.Tensor, num_nodes, lambd):
        c = torch.mm(z1.T, z2)
        c1 = torch.mm(z1.T, z1)
        c2 = torch.mm(z2.T, z2)
        c = c / num_nodes
        c1 = c1 / num_nodes
        c2 = c2 / num_nodes
        loss_inv = -torch.diagonal(c).sum()
        iden = torch.tensor(np.eye(c.shape[0])).cuda()
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()
        loss = loss_inv + lambd * (loss_dec1 + loss_dec2)
        return loss

    def forward(self, edge_index_1, x_1, edge_index_2, x_2):
        h1 = self.backbone(x_1, edge_index_1)
        h2 = self.backbone(x_2, edge_index_2)
        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)
        return z1, z2