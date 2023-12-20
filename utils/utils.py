import torch
from torch_geometric.utils import get_ppr
from torch_geometric.data import Data
import numpy as np

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