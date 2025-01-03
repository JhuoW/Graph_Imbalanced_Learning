#!/bin/bash

# imb_ratio = 10 gnn_backbone=gcn 
python train.py --gpu_id 0 --split imbalance --ssl GRACE --dataset Cora --imb_ratio 10 --multirun 10 --gnn GCN





# for s in {0..9}
# do
#     for m in GCN GAT SAGE GPR
#     do
#     for dataset in amazon-photo cora elliptic ogb-arxiv fb100 twitch-e
#     do
#     python train_both_all.py --seed=$s --gpu_id=0 --model=$m  --tune=0 --dataset=$dataset --debug=1 >> results/${dataset}_${m}_${s}.out
#     done
#     done
# done