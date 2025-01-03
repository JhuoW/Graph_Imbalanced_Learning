import torch
import torch.nn as nn
import torch.nn.functional as F


class Projection(nn.Module):
    def __init__(self, args, in_dim, n_cls):  # in_dim = 128
        super(Projection, self).__init__()
        self.num_proj_layers = args.num_proj_layers
        self.hid_dim         = args.proj_hid_dim
        self.dropout         = args.dropout
        self.lins = nn.ModuleList()
        out_dim  = n_cls if self.num_proj_layers == 1 else self.hid_dim
        
        self.lins.append(nn.Linear(in_dim, out_dim))
        for i in range(self.num_proj_layers - 1):
            out_dim = n_cls if i == self.num_proj_layers - 2 else self.hid_dim
            self.lins.append(nn.Linear(self.hid_dim, out_dim))

    def forward(self, x):
        h = x  # (2708, 128)
        for i, layer in enumerate(self.lins):
            h = layer(h)
            if i != self.num_proj_layers -1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training = self.training)
        return h