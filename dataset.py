from torch_geometric.datasets import CitationFull
import torch_geometric.transforms as T
import torch



import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_planetoid_data


class Planetoid(InMemoryDataset):
    url = 'https://github.com/kimiyoung/planetoid/raw/master/data'
    geom_gcn_url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
                    'geom-gcn/master')

    def __init__(self, root: str, name: str, split: str = "public",   # split: public, imbalance, random
                 imb_ratio: int = 10, fix_minority: bool = True, 
                 num_train_per_class: int = 20, num_val: int = 500,
                 num_test: int = 1000, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name

        self.split = split.lower()
        self.fix_minority = fix_minority
        assert self.split in ['public', 'full', 'geom-gcn', 'random', 'imbalance']

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        if split == 'full':
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])

        elif split == 'random':
            data = self.get(0)
            data.train_mask.fill_(False)
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[remaining[num_val:num_val + num_test]] = True

            self.data, self.slices = self.collate([data])
        elif split == 'imbalance':
            data = self.get(0)
            n_cls = self.num_classes
            self.imb_cls_num = n_cls // 2
            imb_cls_num_list = []
            train_nodes_per_cls, num_train_nodes_per_cls = self.get_idx_info(data, n_cls)
            max_num = np.max(num_train_nodes_per_cls[:n_cls-self.imb_cls_num])
            for i in range(n_cls):
                if imb_ratio > 1 and i > n_cls-1-self.imb_cls_num: # i>3: i = 4,5,6 set to imbalanced class
                    imb_cls_num_list.append(min(int(max_num*(1./imb_ratio)), num_train_nodes_per_cls[i]))
                else:
                    imb_cls_num_list.append(num_train_nodes_per_cls[i])            
            imb_train_mask, imb_nodes_per_cls = self.get_imb_trainset(data, n_cls, data.num_nodes, train_nodes_per_cls,num_train_nodes_per_cls, imb_cls_num_list, fix_minority)
            data.imb_train_mask = imb_train_mask
            self.imb_cls_num_list = imb_cls_num_list
            self.data, self.slices = self.collate([data])
                
    @property
    def raw_dir(self) -> str:
        if self.split == 'geom-gcn':
            return osp.join(self.root, self.name, 'geom-gcn', 'raw')
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        if self.split == 'geom-gcn':
            return osp.join(self.root, self.name, 'geom-gcn', 'processed')
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        return [f'ind.{self.name.lower()}.{name}' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    
    ########## Imbalanced setting ##############
    def get_idx_info(self, data, n_cls):
        train_mask = data.train_mask
        labels = data.y
        index_list = torch.arange(labels.shape[0])  # all node indices
        train_nodes_per_cls = []
        num_train_nodes_per_cls     = []
        for i in range(n_cls):
            cls_indices = index_list[((labels == i) & train_mask)] # all nodes idx with label i
            num_nodes_i = (labels[train_mask] == i).sum()
            train_nodes_per_cls.append(cls_indices)
            num_train_nodes_per_cls.append(int(num_nodes_i.item()))
        return train_nodes_per_cls, num_train_nodes_per_cls

    def get_imb_trainset(self, data, n_cls, n_nodes, train_nodes_per_cls, num_train_nodes_per_cls, class_num_list, fix_minority = True):
        imb_train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        imb_nodes_per_cls = []
        for i in range(n_cls):
            if num_train_nodes_per_cls[i] > class_num_list[i]:
                cls_idx = torch.arange(train_nodes_per_cls[i].shape[0]) if fix_minority else torch.randperm(train_nodes_per_cls[i].shape[0])
                cls_idx = train_nodes_per_cls[i][cls_idx]
                cls_idx = cls_idx[:class_num_list[i]]
                imb_nodes_per_cls.append(cls_idx)
            else:
                imb_nodes_per_cls.append(train_nodes_per_cls[i])
            imb_train_mask[imb_nodes_per_cls[i]] = True
        
        assert imb_train_mask.sum().long() == sum(class_num_list)  # num_nodes in imbalance graph
        assert sum([len(idx) for idx in imb_nodes_per_cls]) == sum(class_num_list)

        return imb_train_mask, imb_nodes_per_cls




    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{name}', self.raw_dir)
        if self.split == 'geom-gcn':
            for i in range(10):
                url = f'{self.geom_gcn_url}/splits/{self.name.lower()}'
                download_url(f'{url}_split_0.6_0.2_{i}.npz', self.raw_dir)

    def process(self):
        data = read_planetoid_data(self.raw_dir, self.name)

        if self.split == 'geom-gcn':
            train_masks, val_masks, test_masks = [], [], []
            for i in range(10):
                name = f'{self.name.lower()}_split_0.6_0.2_{i}.npz'
                splits = np.load(osp.join(self.raw_dir, name))
                train_masks.append(torch.from_numpy(splits['train_mask']))
                val_masks.append(torch.from_numpy(splits['val_mask']))
                test_masks.append(torch.from_numpy(splits['test_mask']))
            data.train_mask = torch.stack(train_masks, dim=1)
            data.val_mask = torch.stack(val_masks, dim=1)
            data.test_mask = torch.stack(test_masks, dim=1)

        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'




# def to_imblanced(data, )