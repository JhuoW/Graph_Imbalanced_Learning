import os.path as osp
import random
from time import perf_counter as t
import yaml
import sys
from yaml import SafeLoader
from args import parse_args
from tqdm import tqdm
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import CitationFull
from torch_geometric.nn import GCNConv
from dataset import *
from SSL.GRACE import Encoder, Model, GATEncoder
from SSL.CCASSG import CCA_SSG
from eval import label_classification
from torch_geometric.utils import add_self_loops
import numpy as np
from balance import balance_embedding_mean_cls, balance_embedding_assign, BalanceMLP, balance_embedding_structure
from sklearn.metrics import f1_score, balanced_accuracy_score
from aug import augment_GRACE, augment_CCA
import datetime
from utils.utils import *
import pathlib
from utils.logger import Logger

METRIC_NAME = ['acc', 'f1', 'bacc']

def get_dataset(path, name, split, imb_ratio, fix_minority):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']
    name = 'dblp' if name == 'DBLP' else name
    if name == 'dblp':
        dataset = CitationFull(path, name, transform = T.NormalizeFeatures())
    else:
        dataset = Planetoid(path, name, split=split, imb_ratio= imb_ratio, fix_minority= fix_minority, transform = T.NormalizeFeatures())

    return dataset

def train(model, x, edge_index):
    model.train()
    optimizer.zero_grad()
    # edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    # edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    # x_1 = drop_feature(x, drop_feature_rate_1)
    # x_2 = drop_feature(x, drop_feature_rate_2)
    if args.ssl == 'GRACE':
        edge_index_1, edge_index_2, x_1, x_2 = augment_GRACE(config, x, edge_index)
        if config['add_self_loops']:
            edge_index_1 = add_self_loops(edge_index_1)[0]
            edge_index_2 = add_self_loops(edge_index_2)[0]
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        loss = model.loss(z1, z2, batch_size=0)
    elif args.ssl == 'CCA-SSG':
        edge_index_1, x_1 = augment_CCA(config, x, edge_index)
        edge_index_2, x_2 = augment_CCA(config, x, edge_index)
        if config['add_self_loops']:
            edge_index_1 = add_self_loops(edge_index_1)[0]
            edge_index_2 = add_self_loops(edge_index_2)[0]
        z1, z2 = model(edge_index_1, x_1, edge_index_2, x_2)
        loss = model.loss(z1, z2, n_nodes, config['lambd'])


    
    loss.backward()
    optimizer.step()

    return loss.item()

def test(args, dataset, data, model: Model, x, edge_index, y, final=False):
    model.eval()
    if args.ssl == 'GRACE':
        z = model(x, edge_index)
    elif args.ssl == 'CCA-SSG':
        z = model.get_embedding(x, edge_index)

    # if args.balanced and args.split == 'imbalance':
    #     if args.balance_type == 'mean_cls':
    #         balanced_data = balance_embedding_mean_cls(dataset, data, z, n_cls, metric=args.similarity_metric)
    #         data = balanced_data
    #     elif args.balance_type == 'assign':
    #         balanced_data = balance_embedding_assign(dataset, data, z, n_cls, metric=args.similarity_metric)
    #         data = balanced_data
    #     elif args.balance_type == 'structure':
    #         balance_data = balance_embedding_structure(dataset, data, z, n_cls, metric=args.similarity_metric)
    #         data = balanced_data

    # data = balanced_data
    
    # print(balanced_data.imb_train_mask.sum())
    
    if args.clf == 'LogReg':
        results = label_classification(args, data, z, ratio=0.1)
        best_test_acc = results['Acc']
        best_test_f1 = results['Macro-F1']
        best_test_bacc = results['BAcc']
        return best_test_acc, best_test_f1, best_test_bacc
    elif args.clf == 'mlp':
        z = z.detach().cpu()
        balanced_mlp = BalanceMLP(config, z.shape[1], n_cls).cuda()
        optimizer_mlp = torch.optim.Adam(balanced_mlp.parameters(), lr=config['BalanceMLP']['lr'], weight_decay=config['BalanceMLP']['weight_decay'])
        loss_func_mlp = nn.CrossEntropyLoss()
        # t =  tqdm(range(config['BalanceMLP']['epochs']), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        t =  tqdm(range(2000), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        if args.split == 'imbalance':
            train_mask = data.imb_train_mask
        elif args.split == 'public':
            train_mask = data.train_mask
            data.new_y = data.y
        # new_y = data.new_y
        y     = data.y
        best_val_f1 = 0
        best_test_acc, best_test_f1,best_test_bacc= 0, 0, 0
        best_val_epoch = -1
        for e in t:
            balanced_mlp.train()
            optimizer_mlp.zero_grad()
            x = z.cuda()
            # new_y = new_y.cuda()
            train_mask = train_mask.cuda()
            logits = balanced_mlp(x)
            loss_mlp = loss_func_mlp(logits[train_mask], y[train_mask])  # 利用伪训练集标签训练分类器
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
                acc_test    = f1_score(y_test, test_preds, average='micro')
                f1_test     = f1_score(y_test, test_preds, average='macro')
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
        return best_test_acc, best_test_f1, best_test_bacc

if __name__ == '__main__':
    args = parse_args()
    logger          = Logger(mode = [print])  
    logger.add_line = lambda : logger.log("-" * 50)
    logger.log(" ".join(sys.argv))
    start_wall_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(osp.join('configs', f'{args.ssl}.yaml')), Loader=SafeLoader)[args.dataset]
    
    if config['use_seed']:
        torch.manual_seed(config['seed'])
    random.seed(12345)


    num_epochs = config['num_epochs']


    path = osp.join('datasets', args.dataset)
    dataset = get_dataset(path, args.dataset, args.split, args.imb_ratio, args.fix_minority)
    data    = dataset[0]
    n_cls   = dataset.num_classes
    n_nodes = data.num_nodes

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.cuda()
    if args.ssl == 'GRACE':
        if args.gnn == 'GCN':
            encoder = Encoder(dataset.num_features, config['hid_dim'], activation=({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']],
                            base_model=({'GCNConv': GCNConv})[config['base_model']], k=config['num_layers']).cuda()
        elif args.gnn == 'GAT':
            encoder = GATEncoder(config, dataset.num_features, config['hid_dim']).cuda()
        model = Model(encoder, config['proj_hidden_dim'], config['proj_hidden_dim'], config['tau']).cuda()
    elif args.ssl == 'CCA-SSG':
        model = CCA_SSG(dataset.num_features, config['hid_dim'], config['out_dim'], n_layers=config['num_layers'], use_mlp=config['use_mlp']).cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    start = t()
    prev = start
    for epoch in range(1, num_epochs + 1):
        loss = train(model, data.x, data.edge_index)

        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

    print("=== Final ===")

    test_accs, test_f1s, test_baccs = [],[],[]
    for r in range(args.multirun):
        if args.split == 'imbalance':
            imb_train_masks = data.imb_train_masks
            imb_train_mask = imb_train_masks[r]
            data.imb_train_mask = imb_train_mask
        best_test_acc, best_test_f1, best_test_bacc = test(args, dataset, data, model, data.x, data.edge_index, data.y, final=True)
        test_accs.append(best_test_acc)
        test_f1s.append(best_test_f1)
        test_baccs.append(best_test_bacc)
    
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
        logger.log("%s: %s" % (METRIC_NAME[i]  , str([round(x,4) for x in lists[i]])))
        logger.log("%s: avg / std = %.4f / %.4f" % (METRIC_NAME[i] , avgs[i] , stds[i]))


# GRACE Cora
# acc: [0.716, 0.732, 0.742, 0.72, 0.721, 0.725, 0.681, 0.733, 0.724, 0.73]                 | 161.01s 
# acc: avg / std = 0.7224 / 0.0164                                                          | 161.01s 
# f1: [0.6517, 0.68, 0.6863, 0.6592, 0.6577, 0.6672, 0.5895, 0.6885, 0.6532, 0.6851]        | 161.01s 
# f1: avg / std = 0.6618 / 0.0292                                                           | 161.01s 
# bacc: [0.6569, 0.6779, 0.6867, 0.66, 0.6652, 0.6672, 0.6137, 0.6851, 0.6571, 0.6789]      | 161.01s 
# bacc: avg / std = 0.6649 / 0.0211                                                         | 161.01s 

# GRACE CiteSeer
# acc: [0.465, 0.533, 0.545, 0.521, 0.4, 0.557, 0.485, 0.412, 0.491, 0.525]                 | 156.89s 
# acc: avg / std = 0.4934 / 0.0540                                                          | 156.89s 
# f1: [0.4545, 0.5354, 0.5311, 0.5267, 0.3965, 0.5533, 0.4794, 0.3851, 0.4761, 0.5236]      | 156.89s 
# f1: avg / std = 0.4862 / 0.0591                                                           | 156.89s 
# bacc: [0.4935, 0.5515, 0.5523, 0.5469, 0.4415, 0.5613, 0.5051, 0.4541, 0.4949, 0.5385]    | 156.89s 
# bacc: avg / std = 0.5140 / 0.0428 

# GRACE PubMed
# acc: [0.668, 0.602, 0.619, 0.642, 0.717, 0.588, 0.626, 0.698, 0.617, 0.686]               | 525.61s 
# acc: avg / std = 0.6463 / 0.0436                                                          | 525.61s 
# f1: [0.6325, 0.5409, 0.5643, 0.5956, 0.6988, 0.5288, 0.5945, 0.6716, 0.5873, 0.6698]      | 525.61s 
# f1: avg / std = 0.6084 / 0.0579                                                           | 525.61s 
# bacc: [0.7108, 0.6599, 0.6748, 0.6937, 0.7521, 0.6497, 0.6787, 0.7396, 0.6737, 0.7278]    | 525.61s 
# bacc: avg / std = 0.6961 / 0.0349                                                         | 525.61s 
        

# CCA-SSG Cora
# acc: [0.733, 0.734, 0.742, 0.718, 0.728, 0.718, 0.686, 0.733, 0.728, 0.71]                | 168.91s 
# acc: avg / std = 0.7230 / 0.0161                                                          | 168.91s 
# f1: [0.6698, 0.6785, 0.6986, 0.6494, 0.6629, 0.6465, 0.6025, 0.6812, 0.6399, 0.6412]      | 168.91s 
# f1: avg / std = 0.6571 / 0.0272                                                           | 168.91s 
# bacc: [0.6723, 0.6761, 0.705, 0.6581, 0.6696, 0.6576, 0.6312, 0.6903, 0.6536, 0.6503]     | 168.91s 
# bacc: avg / std = 0.6664 / 0.0211                                                         | 168.91s 
        
# CCA-SSG CiteSeer
# acc: [0.308, 0.332, 0.303, 0.32, 0.304, 0.306, 0.306, 0.318, 0.306, 0.303]                | 164.76s 
# acc: avg / std = 0.3106 / 0.0096                                                          | 164.76s 
# f1: [0.2188, 0.2274, 0.2156, 0.2234, 0.2164, 0.2191, 0.2169, 0.2223, 0.219, 0.2161]       | 164.76s 
# f1: avg / std = 0.2195 / 0.0038                                                           | 164.76s 
# bacc: [0.3453, 0.3398, 0.3394, 0.35, 0.3404, 0.3447, 0.3409, 0.3482, 0.3422, 0.3394]      | 164.76s 
# bacc: avg / std = 0.3430 / 0.0038                                                         | 164.76s 

# CCA-SSG PubMed
# acc: [0.533, 0.542, 0.542, 0.54, 0.541, 0.532, 0.539, 0.541, 0.537, 0.542]                | 236.86s 
# acc: avg / std = 0.5389 / 0.0037                                                          | 236.86s 
# f1: [0.4714, 0.4742, 0.4752, 0.4726, 0.4732, 0.4696, 0.4726, 0.4741, 0.4706, 0.4756]      | 236.86s 
# f1: avg / std = 0.4729 / 0.0020                                                           | 236.86s 
# bacc: [0.5827, 0.5973, 0.5973, 0.5957, 0.5965, 0.5756, 0.5949, 0.5965, 0.5839, 0.5973]    | 236.86s 
# bacc: avg / std = 0.5918 / 0.0079                                                         | 236.86s 