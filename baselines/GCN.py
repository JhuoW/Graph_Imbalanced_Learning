import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, config, n_feat, n_cls, proj = False, residual = False):
        super(GCN, self).__init__()
        self.n_layers = config['num_layers']
        hid_dim  = config['hid_dim']
        self.dropout = config['dropout']
        self.convs = nn.ModuleList()
        self.proj = proj
        self.residual = residual

        out_dim = hid_dim if proj or residual else n_cls

        for i in range(self.n_layers):
            in_dim = n_feat if i == 0 else hid_dim
            if i == self.n_layers-1:
                hid_dim = out_dim
            self.convs.append(GCNConv(in_channels=in_dim, out_channels=hid_dim))

        if proj:
            self.proj_lin = nn.Linear(in_features=hid_dim, out_features= n_cls)

        # if self.residual:
        #     self.lin1 = nn.Linear(self.n_layers * hid_dim, hid_dim)
        #     self.lin2 = nn.Linear(hid_dim, n_cls)
        
    def forward(self, x, edge_index, edge_weight = None):
        x = F.relu(self.convs[0](x, edge_index))
        xs = [x]
        if self.n_layers > 1:
            for conv in self.convs[1:self.n_layers-1]:
                x = F.relu(conv(x, edge_index))
                xs += [x]
            x = self.convs[-1](x, edge_index)
            xs += [x]


        if self.proj:
            return self.proj_lin(x)
        
        return x
        # if self.residual:
        #     x = self.lin1(xs)


    def __repr__(self):
        return self.__class__.__name__        