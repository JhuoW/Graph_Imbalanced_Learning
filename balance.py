import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.utils import get_ppr, to_networkx
from utils.utils import compute_ppr

def get_idx_info(data, n_cls):
    train_mask = data.train_mask
    labels = data.y
    index_list = torch.arange(labels.shape[0])  # all node indices
    train_nodes_per_cls = []
    num_train_nodes_per_cls     = []
    for i in range(n_cls):
        cls_indices = index_list[((labels == i) & train_mask)] # all nodes idx with label i
        num_nodes_i = (labels[train_mask] == i).sum()
        train_nodes_per_cls.append(cls_indices)
        num_train_nodes_per_cls.append(int(num_nodes_i.item()))
    return train_nodes_per_cls, num_train_nodes_per_cls

def compute_distances(A, B):
    # Squared norms of each row in A and B
    norm_A = np.sum(A**2, axis=1, keepdims=True)
    norm_B = np.sum(B**2, axis=1)

    # Compute distance
    distances = np.sqrt(norm_A + norm_B - 2 * np.dot(A, B.T))
    return distances

def balance_embedding_mean_cls(dataset, data, z, n_cls, metric = 'inner_product'):  
    """
    距离小类训练集均值 inner product 最近的节点作为新的训练集
    """
    x, edge_index = data.x, data.edge_index
    imb_train_mask = data.imb_train_mask.detach().cpu()
    imb_train_idx  = torch.LongTensor(torch.nonzero(imb_train_mask, as_tuple=True)[0])

    imb_cls_num_list = dataset.imb_cls_num_list
    max_num = max(imb_cls_num_list)
    upsamples = np.array(max_num - np.array(imb_cls_num_list)) # [ 0  0  0  0 18 18 18]
    # upsamples = [0,0,0,0,20,20,20]
    z = z.detach().cpu()
    train_cls_mean = scatter_mean(z[imb_train_mask], imb_train_idx, dim = 0).numpy()
    Z = z.detach().cpu().numpy()

    # 先normalize 再计算相似度
    # Z_norm = Z / np.linalg.norm(Z, axis=1, ord=2, keepdims=True) 
    
    # train_cls_mean_norm = train_cls_mean / np.linalg.norm(train_cls_mean, axis=1, ord=2, keepdims=True) 

    if metric == 'inner_product':
        similarity = train_cls_mean @ Z.transpose()  # (n_cls, n_nodes)
    elif metric == 'euclidean':
        similarity = compute_distances(train_cls_mean, Z)

    imb_train_idx = imb_train_idx.numpy()

    similarity[:, imb_train_idx] = 0

    sorted_indices = np.argsort(-similarity, axis=1)  # 与每个类最相似的节点

    # similarity = Z_norm @ Z_norm.transpose()
    # np.fill_diagonal(similarity, 0)
    # similarity[:, imb_train_idx] = 0
    # sorted_indices = np.argsort(-similarity, axis=1) # 排除现有训练集
    new_train_nodes = []
    balanced_data = data.clone()
    new_y = balanced_data.y
    for i in range(n_cls):
        new_train_nodes.extend(sorted_indices[i][: upsamples[i]].tolist())
        new_y[sorted_indices[i][: upsamples[i]].tolist()] = i
    new_train_nodes = np.unique(np.array(new_train_nodes))
    balanced_data.imb_train_mask[new_train_nodes] = True
    balanced_data.new_y = new_y  # 人工标签
    return balanced_data


def balance_embedding_assign(dataset, data, z, n_cls, metric = 'inner_product'):  
    """
    每个小类训练集节点周围取新的training nodes
    """
    x, edge_index = data.x, data.edge_index
    imb_train_mask = data.imb_train_mask.detach().cpu()
    imb_train_idx  = torch.LongTensor(torch.nonzero(imb_train_mask, as_tuple=True)[0])
    imb_train_labels = data.y[imb_train_mask].detach().cpu()

    imb_cls_num_list = dataset.imb_cls_num_list
    max_num = max(imb_cls_num_list)
    upsamples = np.array(max_num - np.array(imb_cls_num_list)) # [ 0  0  0  0 18 18 18]
    # upsamples = [0,0,0,0,20,20,20]
    z = z.detach().cpu()
    
    # Cora: 7-1-3 = 3     label>3: 4,5,6
    # Citeseer: 6-1-3 = 2 lbale>2: 3,4,5
    minority_nodes = imb_train_idx[imb_train_labels > n_cls - 1 - dataset.imb_cls_num].numpy()  # [ 1,  2, 20, 23, 26, 37]
    # print(minority_nodes)  # [ 1,  2, 20, 23, 26, 37]
    # print(data.y[minority_nodes])  # [4, 4, 5, 6, 6, 5]
    
    minority_z     = z[minority_nodes]  # shape (6, 128)  embs of [ 1,  2, 20, 23, 26, 37]
    Z = z.numpy()
    if metric == 'inner_product':
        minority_z_norm = minority_z / np.linalg.norm(minority_z, axis=1, ord=2, keepdims=True)
        Z_norm = Z / np.linalg.norm(Z, axis=1, ord=2, keepdims=True) 
        similarity     = minority_z_norm @ Z_norm.transpose()  # (6, 2708)
    elif metric == 'euclidean':
        similarity = compute_distances(minority_z, Z)        
    imb_train_idx = imb_train_idx.numpy()
    similarity[:, imb_train_idx] = 0 

    minority_labels  = data.y[minority_nodes].detach().cpu().numpy()  # [4 4 5 6 6 5]
    num_per_minority = {}   # 每个小类节点的样本量
    for label in minority_labels:
        num_per_minority[label] = num_per_minority.get(label, 0) + 1

    num_upsample_per_node = {} # 每个类中每个节点要上采样的个数  {4: 9, 5:9, 6:9}
    for i, upsample in enumerate(upsamples):
        if i in num_per_minority.keys():
            upspls_per_node = upsample // num_per_minority[i]
            num_upsample_per_node[i] = upspls_per_node   

    num_upsamples_per_node = np.array([num_upsample_per_node[label] for label in minority_labels])  # [9,9,9,9,9,9] 即similarity 每行选取的节点数
    # print(num_upsamples_per_node.shape)
    sorted_indices = np.argsort(-similarity, axis=1)  # (6, 2708)
    new_train_nodes = []
    balanced_data = data.clone()
    new_y = balanced_data.y
    for i in range(sorted_indices.shape[0]): # 6
        new_train_nodes.extend(sorted_indices[i][: num_upsamples_per_node[i]].tolist())
        # print(sorted_indices[i][: num_upsamples_per_node[i]].tolist())
        new_y[sorted_indices[i][: num_upsamples_per_node[i]].tolist()] = minority_labels[i]

    new_train_nodes = np.unique(np.array(new_train_nodes))
    balanced_data.imb_train_mask[new_train_nodes] = True
    
    balanced_data.new_y = new_y 


    # a = {} 

    # for label in balanced_data.new_y[imb_train_mask].detach().cpu().numpy():
    #     a[label] = a.get(label, 0) + 1
    # print(a)
    return balanced_data



class BalanceMLP(nn.Module):
    def __init__(self, config, in_dim, n_cls):  # in_dim = 128
        super(BalanceMLP, self).__init__()
        self.num_proj_layers = config['BalanceMLP']['num_proj_layers']
        self.hid_dim         = config['BalanceMLP']['proj_hid_dim']
        self.dropout         = config['BalanceMLP']['dropout']
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

# def balance_embedding2(dataset, data, model, n_cls):
#     x, edge_index = data.x, data.edge_index
#     imb_train_mask = data.imb_train_mask.detach().cpu()
#     imb_train_idx  = torch.LongTensor(torch.nonzero(imb_train_mask, as_tuple=True)[0])
#     imb_train_labels = data.y[imb_train_mask].detach().cpu()


#     imb_cls_num_list = dataset.imb_cls_num_list
#     max_num = max(imb_cls_num_list)
#     upsamples = np.array(max_num - np.array(imb_cls_num_list)) # [ 0  0  0  0 18 18 18]
#     upsamples = [0,0,0,0,0,0,0]
#     z = model(x, edge_index).detach().cpu()
#     train_cls_mean = scatter_mean(z[imb_train_mask], imb_train_idx, dim = 0).numpy()
#     Z = z.detach().cpu().numpy()

#     Z_norm = Z / np.linalg.norm(Z, axis=1, ord=2, keepdims=True) 
    
#     # minority_nodes = imb_train_idx[imb_train_labels > n_cls // 2]  # all minority class node ids
#     # minority_feats = 

#     # train_cls_mean_norm = train_cls_mean / np.linalg.norm(train_cls_mean, axis=1, ord=2, keepdims=True) 

#     # similarity = train_cls_mean_norm @ Z_norm.transpose()  # (n_cls, n_nodes)

#     imb_train_idx = imb_train_idx.numpy()

#     similarity[:, imb_train_idx] = 0

#     sorted_indices = np.argsort(-similarity, axis=1)  # 与每个类最相似的节点
    
#     # similarity = Z_norm @ Z_norm.transpose()
#     # np.fill_diagonal(similarity, 0)
#     # similarity[:, imb_train_idx] = 0
#     # sorted_indices = np.argsort(-similarity, axis=1) # 排除现有训练集



#     new_train_nodes = []
#     balanced_data = data.clone()
#     new_y = balanced_data.y
#     for i in range(n_cls):
#         new_train_nodes.extend(sorted_indices[i][: upsamples[i]].tolist())
#         new_y[sorted_indices[i][: upsamples[i]].tolist()] = i
#     new_train_nodes = np.unique(np.array(new_train_nodes))
#     balanced_data.imb_train_mask[new_train_nodes] = True
#     balanced_data.y = new_y
#     return balanced_data


