import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GATConv



class GAT(nn.Module):
    def __init__(self, config, n_feat, n_cls):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList()
        self.n_layers = config['num_layers']
        self.dropout = config['dropout']
        self.hid_dim = config['hid_dim']
        self.head    = config['head']
        self.add_self_loops = config['add_self_loops']
        self.convs.append(GATConv(in_channels = n_feat,
                                  out_channels = self.hid_dim,
                                  heads = self.head,
                                  dropout = self.dropout,
                                  add_self_loops = self.add_self_loops))
        for l in range(1, self.n_layers-1):
            # due to multi-head, the in_dim = num_hidden * n_heads
            self.convs.append(GATConv(in_channels = self.hid_dim * self.head, 
                                      out_channels=self.hid_dim, 
                                      heads=self.head,
                                      dropout = self.dropout,
                                      add_self_loops= self.add_self_loops))
        self.convs.append(GATConv(in_channels = self.hid_dim * self.head,
                                  out_channels = n_cls,
                                  heads = 1,
                                  dropout = self.dropout))
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.n_layers-1):
            x = F.elu(self.convs[i](x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x