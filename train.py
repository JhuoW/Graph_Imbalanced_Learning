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
from balance import balance_embedding_mean_cls, balance_embedding_assign, BalanceMLP
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

    save_embeddings(z.detach().cpu(), path=osp.join('embeddings', args.dataset + '_' + args.ssl + '_' + args.gnn + '_' + str(args.imb_ratio) +'.pt'))

    # if args.split == 'imbalance'
    if args.balanced and args.split == 'imbalance':
        if args.balance_type == 'mean_cls':
            balanced_data = balance_embedding_mean_cls(dataset, data, z, n_cls, metric=args.similarity_metric)
            data = balanced_data
        elif args.balance_type == 'assign':
            balanced_data = balance_embedding_assign(dataset, data, z, n_cls, metric=args.similarity_metric)
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
        if args.split == 'imbalance':
            train_mask = data.imb_train_mask
        elif args.split == 'public':
            train_mask = data.train_mask
            data.new_y = data.y
        new_y = data.new_y
        y     = data.y
        best_val_f1 = 0
        best_test_acc, best_test_f1,best_test_bacc= 0, 0, 0
        best_val_epoch = -1
        for e in t:
            balanced_mlp.train()
            optimizer_mlp.zero_grad()
            x = z.cuda()
            new_y = new_y.cuda()
            train_mask = train_mask.cuda()
            logits = balanced_mlp(x)
            loss_mlp = loss_func_mlp(logits[train_mask], new_y[train_mask])  # 利用伪训练集标签训练分类器
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
    print("saving configs...")
    # save configs
    config_args = args2config(args, config)
    log_dir = osp.join('logs', args.dataset, start_wall_time, f'imb_ratio_{args.imb_ratio}')
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    config_save_file = osp.join(log_dir, '{}.yaml'.format(args.dataset))
    with open(config_save_file, 'w+') as f:
        yaml.dump(config_args, f, sort_keys=True, indent = 2)    

    print("saving model and results...")
    test_result_file = osp.join(log_dir, 'results.txt')
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
    f = open(test_result_file, 'w')
    for i, (avg, std) in enumerate(zip(avgs, stds)):
        logger.log("%s: %s" % (METRIC_NAME[i]  , str([round(x,4) for x in lists[i]])))
        logger.log("%s: avg / std = %.4f / %.4f" % (METRIC_NAME[i] , avgs[i] , stds[i]))

        f.write("%s: %s" % (METRIC_NAME[i]  , str([round(x,4) for x in lists[i]])))
        f.write("\n")
        f.write("%s: avg / std = %.4f / %.4f" % (METRIC_NAME[i] , avgs[i] , stds[i]))
        f.write("\n")
    f.close()

    # with open(test_result_file, 'w') as f:
    #     f.write("%s: %s" % ('acc'  , str([round(x,4) for x in acc_list])))
    #     f.write("%s: avg / std = %.4f / %.4f" % ('acc' , acc_avg , acc_std))
    #     f.write("\n")

    #     f.write("%s: %s" % ('f1'  , str([round(x,4) for x in f1_list])))
    #     f.write("%s: avg / std = %.4f / %.4f" % ('f1' , f1_avg , f1_std))
    #     f.write("\n")     

    #     f.write("%s: %s" % ('bacc'  , str([round(x,4) for x in bacc_list])))
    #     f.write("%s: avg / std = %.4f / %.4f" % ('bacc' , bacc_avg , bacc_std))
    #     f.write("\n")  
    # f.close()

# Imbalanced Grace: Acc=0.7000+-0.0000, Macro-F1=0.6306+-0.0000, BAcc=0.6498+-0.0000
# Balanced Grace: Acc=0.8200+-0.0000, Macro-F1=0.8084+-0.0000, BAcc=0.8267+-0.0000
# Acc=0.7290+-0.0000, Macro-F1=0.6743+-0.0000, BAcc=0.6803+-0.0000 
# (E) | label_classification: Acc=0.6950+-0.0000, Macro-F1=0.6518+-0.0000, BAcc=0.6689+-0.0000


# GraphENS + TAM
# Acc= 0.7124 +- 0.0037, BAcc: 0.6608 +- 0.0050, F1: 0.6554 +- 0.0050
