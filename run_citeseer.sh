#!/bin/bash

python train.py --gpu_id 0 --split imbalance --ssl GRACE --dataset CiteSeer --imb_ratio 10 --multirun 10 --gnn GCN
