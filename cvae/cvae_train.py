import sys
import gc
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
import os.path as osp
from argparse import ArgumentParser
from dataset import Planetoid
import torch_geometric.transforms as T
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from cvae_model import VAE
from projection import Projection
from torch.nn.functional import binary_cross_entropy
import torch.nn as nn
from sklearn.metrics import f1_score, balanced_accuracy_score

def load_embeddings(path):
    return torch.load(path)

def loss_fn(recon_x, x, mean, log_var):
    # Reconstruction Loss
    x = torch.clamp(x, 0, 1)
    recon_x = torch.clamp(recon_x, 0, 1)
    BCE = binary_cross_entropy(recon_x, x, reduction='sum')
    # KLD
    KLD = -0.5  * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return (BCE + KLD) / x.size(0)

def feature_tensor_normalize(feature):
    rowsum = torch.div(1.0, torch.sum(feature, dim=1))
    rowsum[torch.isinf(rowsum)] = 0.
    feature = torch.mm(torch.diag(rowsum), feature)
    return feature

def tune(generated_feats): 
    # the generated model should have high training accuracy

    return 


def get_generator(args, dataset, data, embs, n_cls, mask_id, device):
    mask_id = 7
    imb_cls_num = n_cls // 2
    minor_cls   = np.arange(n_cls)[-imb_cls_num:]  # 4,5,6

    train_masks = data.imb_train_masks
    train_mask  = train_masks[mask_id]
    train_y    = data.y[train_mask].numpy()
    train_idx  = torch.LongTensor(torch.nonzero(train_mask, as_tuple=True)[0]).numpy()
    train_embs = embs[train_mask].numpy()
    
    minor_idx  = train_idx[np.isin(train_y,minor_cls)]  # minority node idx [ 63  91 108 119 123 130]
    minor_y    = data.y[minor_idx].numpy()  # labels of minority nodes [4 4 6 5 6 5]

    feats = []
    conditions = []

    # for i, n in enumerate(minor_idx):
    #     l = minor_y[i]   # label of node n
    #     emb_cls = train_embs[train_y == l]  # condition of node n
    #     feat    = embs[n].numpy()
    #     feats.append(feat)
    #     conditions.append(emb_cls)
    for i, n in enumerate(train_idx):
        l = train_y[i]   # label of node n
        emb_cls = train_embs[train_y == l]  # condition of node n
        feat    = embs[n].numpy()
        feats.append(feat)
        conditions.append(emb_cls)


    feats_x = np.vstack(feats)   # (6, 128)
    
    conditions_x = np.vstack(np.expand_dims(conditions, axis=0))  # shape(6, 2, 128)
    del feats
    del conditions
    gc.collect()


    feats_x = torch.tensor(feats_x, dtype=torch.float32)
    conditions_x = torch.tensor(conditions_x, dtype=torch.float32)

    # cvae_feats  = torch.tensor(conditions_x, dtype=torch.float32)

    cvae_dataset  = TensorDataset(feats_x, conditions_x)
    cvae_dataset_sampler = RandomSampler(cvae_dataset)
    cvae_dataset_loader = DataLoader(cvae_dataset, sampler= cvae_dataset_sampler, batch_size=args.batch_size)

    cvae_model = VAE(encoder_layer_sizes=[embs.shape[1], args.hid_size],
                     latent_size=args.latent_size,
                     decoder_layer_sizes=[args.hid_size, embs.shape[1]],
                     conditional=True,
                     conditional_size=embs.shape[1]).to(device)
    cvae_optimizer = torch.optim.Adam(cvae_model.parameters(), lr=args.cvae_lr)

    projection = Projection(args, embs.shape[1], n_cls).to(device)
    loss_func_proj = nn.CrossEntropyLoss()
    optimizer_proj = torch.optim.Adam(projection.parameters(), lr=args.proj_lr, weight_decay=args.proj_weight_decay)

    # number of upsamples of each classes 
    imb_cls_num_list = dataset.imb_cls_num_list  # [20, 20, 20, 20, 2, 2, 2]
    max_num = max(imb_cls_num_list)
    upsamples = np.array(max_num - np.array(imb_cls_num_list))  # number of upsamples for each classes  [0,0,0,0 18, 18, 18]
    if args.upsample == 'induce':
        upsamples_minor = upsamples[minor_cls]  # [18, 18, 18]
    else:
        upsamples_minor = [args.num_upsample] * imb_cls_num  # [50, 50, 50]

    condition_per_minor = []
    for l in minor_cls:
        emb_cls = train_embs[train_y == l]
        emb_cls = torch.tensor(emb_cls, dtype = torch.float32)
        condition_per_minor.append(emb_cls)

    for epoch in range(args.pretrain_epochs):
        for _, (x, condition) in enumerate(cvae_dataset_loader):
            cvae_model.train()
            x, condition = x.to(device).squeeze(), condition.to(device).squeeze()

            recon_x, mean, log_var, _ = cvae_model(x, condition)

            cvae_loss = loss_fn(recon_x, x, mean, log_var)
            cvae_optimizer.zero_grad()
            cvae_loss.backward()
            cvae_optimizer.step()
            
            # z = torch.randn([sum(upsamples_minor), args.latent_size]).to(device)
            

        if epoch > args.pretrain_epochs // 2:
            # 为每个condition 生成节点
            z = torch.randn([sum(upsamples_minor), args.latent_size]).to(device)  # (54, 10)  
            z_split = torch.split(z, np.array(upsamples_minor).tolist())
            generated_list = []
            generated_label = []
            for condition_id, z_list in enumerate(z_split): # z for each condition
                # z together with ocndition to generate x
                label = minor_cls[condition_id]
                condition = condition_per_minor[condition_id].to(device)
                for z in z_list:
                    generated_x = cvae_model.inference(z, condition)
                    generated_list.append(generated_x)
                    generated_label.append(label)
            generated_features = torch.stack(generated_list).to(device)   # (54, 128)  3个小类 每个小类生成18个节点
            generated_features = feature_tensor_normalize(generated_features)
            generated_labels   = torch.tensor(generated_label, dtype=torch.long).to(device)  # (54)

            extend_features = torch.cat((embs.to(device), generated_features), dim = 0)  # (2762, 128)
            extend_labels   = torch.cat((data.y.to(device), generated_labels), dim = 0)  # (2762)
            extend_train_mask = torch.cat((train_mask.to(device), torch.tensor([True]).repeat(generated_labels.shape[0]).to(device))) # (2762)
            extend_val_mask   = torch.cat((data.val_mask.to(device), torch.tensor([False]).repeat(generated_labels.shape[0]).to(device))) 
            extend_test_mask   = torch.cat((data.test_mask.to(device), torch.tensor([False]).repeat(generated_labels.shape[0]).to(device))) 

            best_val_f1 = 0
            best_test_acc, best_test_f1,best_test_bacc= 0, 0, 0
            best_val_epoch = -1
            for e in range(args.proj_epoch):            
                projection.train()
                optimizer_proj.zero_grad()
                logits = projection(extend_features)
                loss_proj = loss_func_proj(logits[extend_train_mask], extend_labels[extend_train_mask])
                loss_proj.backward(retain_graph=True)
                optimizer_proj.step()
                with torch.no_grad():
                    projection.eval()
                    extend_val_mask = extend_val_mask.cuda()
                    extend_test_mask = extend_test_mask.cuda()
                    preds = projection(extend_features).argmax(dim = -1)
                    y_val = extend_labels[extend_val_mask].detach().cpu().numpy()
                    y_test = extend_labels[extend_test_mask].detach().cpu().numpy()
                    val_preds = preds[extend_val_mask].detach().cpu().numpy()
                    test_preds  = preds[extend_test_mask].detach().cpu().numpy()    

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
                    print("[Epoch] %d [Test Acc] %.4f [Test F1] %.4f [Test bacc] %.4f" % (e, best_test_acc, best_test_f1, best_test_bacc))





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--split', type = str, default='imbalance')  # imbalance, public, random
    parser.add_argument('--imb_ratio', type = int, default=10)
    parser.add_argument('--gnn', type = str, default='GCN')     # 
    parser.add_argument('--ssl', type = str, default='GRACE')  # GRACE CCA-SSG
    parser.add_argument('--multirun', type=int, default=1)
    parser.add_argument('--batch_size', type = int, default = 1)
    ####### VAE Parameters #######
    parser.add_argument('--hid_size', type = int, default= 64)
    parser.add_argument('--latent_size', type = int, default = 64)
    parser.add_argument('--cvae_lr', type = float, default=0.01)
    parser.add_argument('--pretrain_epochs', type = int, default=50)
    parser.add_argument('--upsample', type = str, default='induce')
    parser.add_argument('--num_upsample', type = int, default=3)
    parser.add_argument('--start_IB', type=int, default=5)
    ###### Project Config #######
    parser.add_argument('--num_proj_layers', type = int, default=1)
    parser.add_argument('--proj_hid_dim', type = int, default=64)
    parser.add_argument('--dropout',  type = float, default=0.5)
    parser.add_argument('--proj_epoch', type = int, default=500)
    parser.add_argument('--proj_lr', type = float, default=0.01)
    parser.add_argument('--proj_weight_decay', type = float, default=0.000005)

    args = parser.parse_args()
    path = osp.join('datasets', args.dataset)
    dataset = Planetoid(path, args.dataset, split=args.split, imb_ratio= args.imb_ratio, transform = T.NormalizeFeatures())
    data = dataset[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z = load_embeddings(osp.join('embeddings', args.dataset + '_' + args.ssl + '_' + args.gnn + '_' + str(args.imb_ratio) +'.pt'))
    n_cls = dataset.num_classes

    for run_id in range(args.multirun):
        get_generator(args, dataset, data, z, n_cls, run_id, device)