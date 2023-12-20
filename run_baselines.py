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
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from collections import namedtuple
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
from utils.utils import get_weight, pc_softmax

METRIC_NAME = ['acc', 'f1', 'bacc']
def get_dataset(path, name, split, imb_ratio, fix_minority, config):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']
    name = 'dblp' if name == 'DBLP' else name
    if name == 'dblp':
        dataset = CitationFull(path, name, transform = T.NormalizeFeatures())
    else:
        dataset = Planetoid(path, name, split=split, imb_ratio= imb_ratio, fix_minority= fix_minority, transform = T.NormalizeFeatures() if config['feat_norm'] else None)

    return dataset 

def train(args, model, data, optimizer, loss_func):
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
    if args.model in ['GCN', 'GAT','GCN-RW', 'GAT-RW']:
        logits = model(x, edge_index)
        loss   = loss_func(logits[train_mask], labels[train_mask])

        loss.backward()
        optimizer.step()
    return model, loss.item()

def evaluate(args, model, data, n_data = None):
    model.eval()
    x = data.x.cuda()
    edge_index = data.edge_index.cuda()
    logits = model(x, edge_index)
    if args.pc:
        logits = pc_softmax(logits, n_data)

    val_mask = data.val_mask.detach().cpu()
    test_mask = data.test_mask.detach().cpu()
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

    acc_score_test = int((test_pred == test_y).sum()) / int(test_mask.sum())
    
    DataType   = namedtuple('Metrics', ['val_acc'  , 'val_f1'   , 'val_bacc',
                                        'test_acc' , 'test_f1'  , 'test_bacc', 'acc_score_test'])
    results    = DataType(val_acc   = val_acc, val_f1   = val_f1, val_bacc   = val_bacc,
                          test_acc  = test_acc, test_f1  = test_f1, test_bacc  = test_bacc, acc_score_test = acc_score_test)    

    return results

def main(run_id):
    if args.model in ['GCN', 'GCN-RW']:
        model = GCN(config, n_feat, n_cls, proj=config['proj']).cuda()
    elif args.model in ['GAT', 'GAT-RW']:
        model = GAT(config, n_feat, n_cls).cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    n_data = None
    if args.model in ['GCN-RW', 'GAT-RW', 'GCN', 'GAT']:
        n_data = [] # 存放每个类的训练集节点个数
        stats = data.y[data.imb_train_masks[run_id]] # training labels
        for i in range(n_cls):
            data_num = (stats == i).sum()  # number of training nodes with label i
            n_data.append(int(data_num.item()))
        class_weight = get_weight(True, n_data).cuda()


    loss_func = nn.CrossEntropyLoss(weight=class_weight if args.model in ['GCN-RW', 'GAT-RW'] else None)

    best_val_metric  = 0

    best_metric_epoch 	= -1 # best number on dev set
    patience_cnt 		= 0
    best_model          = None
    best_results        = None
    best_classifier     = None
    monitor             = config['monitor']
    data.imb_train_mask = data.imb_train_masks[run_id]
    t = tqdm(range(1, num_epochs + 1), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for epoch in t:
        model, loss = train(args, model, data, optimizer, loss_func)
        with torch.no_grad():
            results = evaluate(args, model, data, n_data)
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
            postfix_str = 'Epoch=%d, loss=%.4f, val_%s = %.4f, best_val_%s = %.4f,| Test acc = %.4f, f1 = %.4f, bacc = %.4f, acc_score = %.4f' %(
                epoch, loss, monitor, val_metric, monitor, best_val_metric, best_results.test_acc, best_results.test_f1, best_results.test_bacc, best_results.acc_score_test
            )
              
            t.set_postfix_str(postfix_str)
    print("=== Final ===")
    print(f'Run {run_id} Test acc = {best_results.test_acc:.4f}, f1 = {best_results.test_f1:.4f}, bacc = {best_results.test_bacc:.4f}, acc_score = {best_results.acc_score_test:.4f}')
    return best_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CiteSeer')
    parser.add_argument('--model', type = str, default='GCN')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--fix_minority', type=bool, default= True)
    parser.add_argument('--imb_ratio', type = int, default=10)
    parser.add_argument('--split', type = str, default='imbalance')  # imbalance, public, random
    parser.add_argument('--multirun', type = int, default= 10)
    parser.add_argument('--pc', action='store_true', default= False)
    args = parser.parse_args()
    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)
    # torch.manual_seed(1234567)
    config = yaml.load(open(osp.join('configs', '{}.yaml'.format(args.model))), Loader=SafeLoader)[args.dataset]

    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    num_epochs = config['epochs']

    path = osp.join('datasets', args.dataset)
    dataset = get_dataset(path, args.dataset, args.split, args.imb_ratio, args.fix_minority, config)
    data    = dataset[0]

    if osp.exists(f'datasets/{args.dataset}/{args.dataset}/processed/{args.dataset}_{args.imb_ratio}.pt'):
        print('Using existing masks')
        train_masks = torch.load(f'datasets/{args.dataset}/{args.dataset}/processed/{args.dataset}_{args.imb_ratio}.pt')
        data.imb_train_masks = train_masks
    else:
        raise ValueError

    if config['add_self_loops']:
        data.edge_index = add_self_loops(data.edge_index)[0]
    n_cls   = dataset.num_classes
    n_nodes = data.num_nodes
    n_feat  = data.x.shape[1]

    test_accs, test_f1s, test_baccs = [],[],[]
    for i in range(args.multirun):
        
        best_results = main(i)
        test_accs.append(best_results.test_acc)
        test_f1s.append(best_results.test_f1)
        test_baccs.append(best_results.test_bacc)

    acc_list = np.around(test_accs, decimals=5)
    acc_avg = np.mean(acc_list, axis=0)
    acc_std = np.std(acc_list, axis=0, ddof=1)
        
    f1_list = np.around(test_f1s, decimals=5)
    f1_avg = np.mean(f1_list, axis=0)
    f1_std = np.std(f1_list, axis=0, ddof=1)

    bacc_list = np.around(test_baccs, decimals=5)
    bacc_avg = np.mean(bacc_list, axis=0)
    bacc_std = np.std(bacc_list, axis=0, ddof=1)
    lists = [acc_list,f1_list,bacc_list]
    avgs = [acc_avg, f1_avg, bacc_avg]
    stds = [acc_std, f1_std, bacc_std]
    for i, (avg, std) in enumerate(zip(avgs, stds)):
        print("%s: %s" % (METRIC_NAME[i]  , str([round(x,4) for x in lists[i]])))
        print("%s: avg / std = %.4f / %.4f" % (METRIC_NAME[i] , avgs[i] , stds[i]))

# GCN-RW Cora python run_baselines.py --dataset Cora --model GCN-RW --pc
# acc: [0.738, 0.743, 0.751, 0.746, 0.747, 0.752, 0.746, 0.727, 0.729, 0.726]
# acc: avg / std = 0.7405 / 0.0099
# f1: [0.6867, 0.6962, 0.7249, 0.7003, 0.7165, 0.7143, 0.7252, 0.7083, 0.6971, 0.7039]
# f1: avg / std = 0.7073 / 0.0128
# bacc: [0.6946, 0.6948, 0.7345, 0.7099, 0.7215, 0.7176, 0.7248, 0.7323, 0.6926, 0.7087]
# bacc: avg / std = 0.7131 / 0.0156


# GCN Cora Pc
# acc: [0.729, 0.741, 0.747, 0.712, 0.73, 0.723, 0.723, 0.736, 0.703, 0.711]
# acc: avg / std = 0.7255 / 0.0140
# f1: [0.68, 0.6939, 0.7224, 0.6737, 0.6775, 0.6714, 0.6837, 0.7089, 0.6744, 0.6793]
# f1: avg / std = 0.6865 / 0.0169
# bacc: [0.6801, 0.6898, 0.7293, 0.6955, 0.6762, 0.6703, 0.6866, 0.7244, 0.685, 0.6805]
# bacc: avg / std = 0.6918 / 0.0198

# GCN CiteSeer PC
# acc: [0.407, 0.437, 0.314, 0.391, 0.354, 0.483, 0.478, 0.347, 0.443, 0.42]
# acc: avg / std = 0.4074 / 0.0563
# f1: [0.3823, 0.398, 0.2854, 0.344, 0.3276, 0.4503, 0.4394, 0.3008, 0.4105, 0.3972]
# f1: avg / std = 0.3735 / 0.0566
# bacc: [0.4371, 0.454, 0.3475, 0.4143, 0.3428, 0.5081, 0.4986, 0.3655, 0.463, 0.4072]
# bacc: avg / std = 0.4238 / 0.0591        
        
# GCN PubMed PC
# acc: [0.71, 0.64, 0.686, 0.695, 0.674, 0.636, 0.717, 0.71, 0.684, 0.737]
# acc: avg / std = 0.6889 / 0.0324
# f1: [0.7183, 0.6278, 0.6869, 0.6857, 0.6896, 0.6326, 0.7189, 0.7075, 0.68, 0.736]
# f1: avg / std = 0.6883 / 0.0355
# bacc: [0.7169, 0.6798, 0.7073, 0.7282, 0.6955, 0.6836, 0.7319, 0.726, 0.7186, 0.7574]
# bacc: avg / std = 0.7145 / 0.0237
        
# GAT Cora PC
# acc: [0.696, 0.704, 0.74, 0.718, 0.648, 0.678, 0.705, 0.722, 0.66, 0.693]
# acc: avg / std = 0.6964 / 0.0282
# f1: [0.6524, 0.6497, 0.6964, 0.675, 0.6521, 0.6534, 0.6694, 0.695, 0.5814, 0.6519]
# f1: avg / std = 0.6577 / 0.0322
# bacc: [0.6774, 0.6491, 0.6951, 0.6846, 0.6632, 0.6643, 0.668, 0.7088, 0.5946, 0.6642]
# bacc: avg / std = 0.6669 / 0.0308
    
# GAT CiteSeer PC
# acc: [0.451, 0.451, 0.374, 0.377, 0.376, 0.385, 0.486, 0.446, 0.451, 0.593]
# acc: avg / std = 0.4390 / 0.0678
# f1: [0.4278, 0.4081, 0.3016, 0.3503, 0.3576, 0.3418, 0.4361, 0.392, 0.4201, 0.5652]
# f1: avg / std = 0.4001 / 0.0725
# bacc: [0.4665, 0.4413, 0.3812, 0.4112, 0.41, 0.4096, 0.4779, 0.4187, 0.4653, 0.5642]
# bacc: avg / std = 0.4446 / 0.0522

# GAT PubMed PC
# acc: [0.738, 0.672, 0.7, 0.731, 0.719, 0.629, 0.708, 0.757, 0.66, 0.747]
# acc: avg / std = 0.7061 / 0.0413
# f1: [0.7369, 0.6676, 0.7036, 0.7258, 0.7139, 0.6278, 0.7005, 0.7522, 0.6555, 0.747]
# f1: avg / std = 0.7031 / 0.0412
# bacc: [0.7417, 0.7022, 0.7108, 0.7286, 0.7377, 0.6812, 0.7295, 0.7728, 0.7035, 0.7524]
# bacc: avg / std = 0.7261 / 0.0271