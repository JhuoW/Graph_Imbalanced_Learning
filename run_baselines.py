from torch_geometric.datasets import CitationFull
import argparse
import torch
from time import perf_counter as t
import yaml
from yaml import SafeLoader
import os.path as osp
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from baselines.GCN import GCN
from baselines.GAT import GAT
from dataset import *
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score, balanced_accuracy_score
from collections import namedtuple


def get_dataset(path, name, split, imb_ratio, fix_minority):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']
    name = 'dblp' if name == 'DBLP' else name
    if name == 'dblp':
        dataset = CitationFull(path, name, transform = T.NormalizeFeatures())
    else:
        dataset = Planetoid(path, name, split=split, imb_ratio= imb_ratio, fix_minority= fix_minority, transform = T.NormalizeFeatures())

    return dataset 

def train(args, model, data, loss_func):
    model.train()
    optimizer.zero_grad()
    x = data.x.cuda()
    edge_index = data.edge_index.cuda()
    if args.split == 'public':
        train_mask = data.train_mask.cuda()
    elif args.split == 'imbalance':
        imb_train_mask = data.imb_train_mask.cuda()
        train_mask = imb_train_mask
    labels = data.y.cuda()
    if args.model in ['GCN', 'GAT']:
        logits = model(x, edge_index)
        loss   = loss_func(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
    return model, loss.item()

def evaluate(model, data):
    model.eval()
    x = data.x.cuda()
    edge_index = data.edge_index.cuda()
    logits = model(x, edge_index)
    val_mask = data.val_mask.cuda()
    test_mask = data.test_mask.cuda()
    labels = data.y

    val_pred = torch.nn.Softmax(dim=1)(logits.detach().cpu())[val_mask]
    val_pred = np.argmax(val_pred.numpy(), axis=1)
    val_y    = labels[val_mask].detach().cpu().numpy()

    val_f1 = f1_score(val_y, val_pred, average='macro')
    val_acc = f1_score(val_y, val_pred, average='micro')
    val_bacc = balanced_accuracy_score(val_y, val_pred)

    test_pred = torch.nn.Softmax(dim=1)(logits.detach().cpu())[test_mask]
    test_pred = np.argmax(test_pred.numpy(), axis=1)
    test_y    = labels[test_mask].detach().cpu().numpy()

    test_f1 = f1_score(test_y, test_pred, average='macro')
    test_acc = f1_score(test_y, test_pred, average='micro')
    test_bacc = balanced_accuracy_score(test_y, test_pred)

    DataType   = namedtuple('Metrics', ['val_acc'  , 'val_f1'   , 'val_bacc',
                                        'test_acc' , 'test_f1'  , 'test_bacc'])
    results    = DataType(val_acc   = val_acc, val_f1   = val_f1, val_bacc   = val_bacc,
                          test_acc  = test_acc, test_f1  = test_f1, test_bacc  = test_bacc)    

    return results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--model', type = str, default='GAT')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--fix_minority', type=bool, default= True)
    parser.add_argument('--imb_ratio', type = int, default=10)
    parser.add_argument('--split', type = str, default='imbalance')  # imbalance, public, random
    args = parser.parse_args()
    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(osp.join('configs', '{}.yaml'.format(args.model))), Loader=SafeLoader)[args.dataset]

    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    num_epochs = config['epochs']

    path = osp.join('datasets', args.dataset)
    dataset = get_dataset(path, args.dataset, args.split, args.imb_ratio, args.fix_minority)
    data    = dataset[0]
    n_cls   = dataset.num_classes
    n_nodes = data.num_nodes
    n_feat  = data.x.shape[1]
    
    if args.model == 'GCN':
        model = GCN(config, n_feat, n_cls, proj=config['proj']).cuda()
    elif args.model == 'GAT':
        model = GAT(config, n_feat, n_cls).cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    loss_func = nn.CrossEntropyLoss()

    best_val_metric  = 0

    best_metric_epoch 	= -1 # best number on dev set
    patience_cnt 		= 0
    best_model          = None
    best_results        = None
    best_classifier     = None
    monitor             = config['monitor']
    
    for epoch in range(1, num_epochs + 1):
        model, loss = train(args, model, data, loss_func)
        with torch.no_grad():
            results = evaluate(model, data)
            val_metric = getattr(results, 'val_{}'.format(monitor))
            if val_metric >= best_val_metric:
                best_metric_epoch 	= epoch
                best_val_metric     = val_metric
                best_results        = results
                best_model          = model
                patience_cnt        = 0
            else:
                patience_cnt     +=1
            if config['patience'] > 0 and patience_cnt >= config['patience']:
                break
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, val_{monitor} = {val_metric:.4f}, best_val_{monitor} = {best_val_metric:.4f},',
              f'| Test acc = {best_results.test_acc:.4f}, f1 = {best_results.test_f1}, bacc = {best_results.test_bacc}')
    print("=== Final ===")
    print(f'Test acc = {best_results.test_acc:.4f}, f1 = {best_results.test_f1:.4f}, bacc = {best_results.test_bacc:.4f}')