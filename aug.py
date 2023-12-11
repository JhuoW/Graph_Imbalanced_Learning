from torch_geometric.utils import dropout_edge
import torch


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def augment_GRACE(config, x, edge_index):
    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    edge_index_1 = dropout_edge(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_edge(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    return edge_index_1, edge_index_2, x_1, x_2
    

def augment_CCA(config, x, edge_index):
    drop_edge_rate = config['der']
    drop_feature_rate = config['dfr']    
    edge_index = dropout_edge(edge_index, p=drop_edge_rate)[0]
    x = drop_feature(x, drop_feature_rate)
    return edge_index, x