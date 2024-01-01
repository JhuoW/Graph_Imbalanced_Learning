import argparse


# python train.py --split imbalance --ssl GRACE --dataset CiteSeer
# python train.py --split imbalance --ssl GRACE --gnn GAT  --dataset CiteSeer --multirun 10
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--split', type = str, default='imbalance')  # imbalance, public, random
    parser.add_argument('--ssl', type = str, default='GRACE')  # GRACE CCA-SSG
    parser.add_argument('--gnn', type = str, default='GCN')     # 
    parser.add_argument('--balanced',  type=bool, default= True)
    # parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--imb_ratio', type = int, default=10)
    parser.add_argument('--fix_minority', type=bool, default= True)
    parser.add_argument('--clf', type = str, default='mlp')   # LogReg, mlp   
    parser.add_argument('--similarity_metric', type = str, default='inner_product')  # inner_product, euclidean
    parser.add_argument('--balance_type', type = str, default='assign')            # mean_cls, assign, structure
    parser.add_argument('--multirun', type = int, default=10)
    args = parser.parse_args()

    return args