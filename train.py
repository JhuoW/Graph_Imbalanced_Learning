import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
from args import parse_args
from tqdm import tqdm
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv
from dataset import *
from SSL.GRACE import Encoder, Model, drop_feature
from eval import label_classification
import numpy as np
from balance import balance_embedding_mean_cls, balance_embedding_assign, BalanceMLP
from sklearn.metrics import f1_score, balanced_accuracy_score


def get_dataset(path, name, split, imb_ratio, fix_minority):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']
    name = 'dblp' if name == 'DBLP' else name
    if name == 'dblp':
        dataset = CitationFull(path, name, transform = T.NormalizeFeatures())
    else:
        dataset = Planetoid(path, name, split=split, imb_ratio= imb_ratio, fix_minority= fix_minority, transform = T.NormalizeFeatures())

    return dataset 

def train(model: Model, x, edge_index):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(args, dataset, data, model: Model, x, edge_index, y, final=False):
    model.eval()
    z = model(x, edge_index)

    # if args.split == 'imbalance'
    if args.balanced:
        if args.balance_type == 'mean_cls':
            balanced_data = balance_embedding_mean_cls(dataset, data, model, n_cls, metric=args.similarity_metric)
            data = balanced_data
        elif args.balance_type == 'assign':
            balanced_data = balance_embedding_assign(dataset, data, model, n_cls, metric=args.similarity_metric)
            data = balanced_data


    # data = balanced_data
    
    # print(balanced_data.imb_train_mask.sum())
    
    if args.clf == 'LogReg':
        label_classification(args, data, z, ratio=0.1)
    elif args.clf == 'mlp':
        z = z.detach().cpu()
        balanced_mlp = BalanceMLP(config, z.shape[1], n_cls).cuda()
        optimizer_mlp = torch.optim.Adam(balanced_mlp.parameters(), lr=config['BalanceMLP']['lr'], weight_decay=config['BalanceMLP']['weight_decay'])
        loss_func_mlp = nn.CrossEntropyLoss()
        t =  tqdm(range(config['BalanceMLP']['epochs']), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        train_mask = data.imb_train_mask
        new_y = data.new_y
        y     = data.y
        best_val_f1 = 0
        best_test_acc, best_test_f1,best_test_bacc= 0,0,0
        best_val_epoch = -1
        for e in t:
            balanced_mlp.train()
            optimizer_mlp.zero_grad()
            x = z.cuda()
            new_y = new_y.cuda()
            train_mask = train_mask.cuda()
            logits = balanced_mlp(x)
            loss_mlp = loss_func_mlp(logits[train_mask], new_y[train_mask])
            loss_mlp.backward()
            optimizer_mlp.step()
            with torch.no_grad():
                balanced_mlp.eval()
                val_mask  = data.val_mask.cuda()
                y         = data.y.cuda()
                test_mask = data.test_mask.cuda()
                preds = balanced_mlp(x).argmax(dim = -1)

                y_val = y[val_mask].detach().cpu().numpy()
                y_test = y[test_mask].detach().cpu().numpy()
                val_preds = preds[val_mask].detach().cpu().numpy()
                test_preds  = preds[test_mask].detach().cpu().numpy()

                acc_val = f1_score(y_val, val_preds, average='micro')
                f1_val      = f1_score(y_val, val_preds, average='macro')
                bacc_val    = balanced_accuracy_score(y_val, val_preds)
                acc_test = f1_score(y_test, test_preds, average='micro')
                f1_test   = f1_score(y_test, test_preds, average='macro')
                bacc_test    = balanced_accuracy_score(y_test, test_preds)
                if f1_val >= best_val_f1:
                    best_val_epoch = e
                    best_val_f1 = f1_val
                    best_test_acc = acc_test
                    best_test_f1  = f1_test
                    best_test_bacc = bacc_test
            postfix_str = "<Epoch %d> [Val Acc] %.4f [Val F1] %.4f [Val bacc] %.4f <Best Val:> [Epoch] %d <Test:> [Test Acc] %.4f [Test F1] %.4f [Test bacc] %.4f" % (
                epoch, acc_val,f1_val,bacc_val, best_val_epoch, best_test_acc, best_test_f1, best_test_bacc
            )
            t.set_postfix_str(postfix_str)
        print("[Test Acc] %.4f [Test F1] %.4f [Test bacc] %.4f" % (best_test_acc, best_test_f1, best_test_bacc)) 


if __name__ == '__main__':
    args = parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(osp.join('configs', 'GRACE.yaml')), Loader=SafeLoader)[args.dataset]

    torch.manual_seed(config['seed'])
    random.seed(12345)

    learning_rate = config['learning_rate']
    num_hidden = config['hid_dim']
    num_proj_hidden = config['proj_hidden_dim']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    # fix_minority = config['fix_minority']

    path = osp.join('datasets', args.dataset)
    dataset = get_dataset(path, args.dataset, args.split, args.imb_ratio, args.fix_minority)
    data    = dataset[0]
    n_cls   = dataset.num_classes
    n_nodes = data.num_nodes

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.cuda()

    encoder = Encoder(dataset.num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers).cuda()
    model = Model(encoder, num_hidden, num_proj_hidden, tau).cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start
    for epoch in range(1, num_epochs + 1):
        loss = train(model, data.x, data.edge_index)

        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

    print("=== Final ===")
    test(args, dataset, data, model, data.x, data.edge_index, data.y, final=True)

# Imbalanced Grace: Acc=0.7000+-0.0000, Macro-F1=0.6306+-0.0000, BAcc=0.6498+-0.0000
# Balanced Grace: Acc=0.8200+-0.0000, Macro-F1=0.8084+-0.0000, BAcc=0.8267+-0.0000
# Acc=0.7290+-0.0000, Macro-F1=0.6743+-0.0000, BAcc=0.6803+-0.0000
# (E) | label_classification: Acc=0.6950+-0.0000, Macro-F1=0.6518+-0.0000, BAcc=0.6689+-0.0000


# GraphENS + TAM
# Acc= 0.7124 +- 0.0037, BAcc: 0.6608 +- 0.0050, F1: 0.6554 +- 0.0050
