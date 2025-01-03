import torch
from torch_geometric.utils import get_ppr
from torch_geometric.data import Data
import numpy as np
import networkx as nx
from scipy.linalg import fractional_matrix_power, inv

def args2config(args, config:dict):
    args_ = vars(args)
    config.update(args_)
    return config

def save_embeddings(embs, path):
    torch.save(embs, path)
    print("Saved.")

def load_embeddings(path):
    return torch.load(path)

def compute_ppr(graph: Data, nodes):
    edge_index, weight = get_ppr(graph.edge_index, nodes)

def get_weight(is_reweight, class_num_list):
    if is_reweight:
        min_number = np.min(class_num_list)
        class_weight_list = [float(min_number)/float(num) for num in class_num_list]
    else:
        class_weight_list = [1. for _ in class_num_list]
    class_weight = torch.tensor(class_weight_list).type(torch.float32)

    return class_weight


def pc_softmax(logits, cls_num):
    sample_per_class = torch.tensor(cls_num)
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits - spc.log()
    return logits


def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1

