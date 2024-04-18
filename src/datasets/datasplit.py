# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import is_undirected

import settings
from utils import logger
from utils import set_seed


class DataSplit:
    def __init__(
            self,
            dataset_name: str,
            num_splits=5
    ):
        self.dataset_name = dataset_name
        self._data: Optional[Data] = None
        self._dataset = None
        self.num_splits = num_splits

    @property
    def data(self):
        if self._data is None:
            from datasets.dataloader import load_dataset
            self._dataset = load_dataset(self.dataset_name)
            self._data = self._dataset[0]
        return self._data

    @property
    def node_split_root(self):
        return settings.DATASPLIT_ROOT / self.dataset_name / f"node-{self.num_splits}splits"

    def node_split_paths(self, split_i, mkdir=False) -> Dict[str, Path]:
        if mkdir:
            self.node_split_root.mkdir(parents=True, exist_ok=True)

        return {
            'train_node_index': self.node_split_root / f"train-node-index-split{split_i}.pt",
            'val_node_index': self.node_split_root / f"val-node-index-split{split_i}.pt",
            'test_node_index': self.node_split_root / f"test-node-index-split{split_i}.pt"
        }

    def load_node_split(self, split_i):
        split_paths = self.node_split_paths(split_i)
        if not all([p.exists() for p in split_paths.values()]):
            self.generate_node_splits()

        return {key: torch.load(path).long() for key, path in split_paths.items()}

    def generate_node_splits(self):
        """include nodes with labels in trainin/validation/testing for node classification task"""
        labeled_node_mask = self.data.y >= 0
        assert -1 not in torch.unique(self.data.y[labeled_node_mask]).tolist()

        node_mask = labeled_node_mask.cpu().numpy()
        target_nodes = np.nonzero(node_mask)[0]
        assert len(target_nodes) == self.data.x.shape[0]

        """k-fold splits"""
        kf = KFold(n_splits=self.num_splits, random_state=101, shuffle=True)
        splits = kf.split(np.arange(len(target_nodes)))

        for split_i, (train_index, test_index) in enumerate(splits):
            # use 20% of training data for validation
            train_index, val_index = \
                train_test_split(train_index, test_size=0.2, random_state=split_i * 127, shuffle=True)

            split_paths = self.node_split_paths(split_i, mkdir=True)
            torch.save(torch.from_numpy(target_nodes[np.sort(train_index)]).int(), split_paths['train_node_index'])
            torch.save(torch.from_numpy(target_nodes[np.sort(val_index)]).int(), split_paths['val_node_index'])
            torch.save(torch.from_numpy(target_nodes[np.sort(test_index)]).int(), split_paths['test_node_index'])

        logger.info(f"node splits generated for {self.dataset_name}")

    @property
    def edge_split_root(self):
        return settings.DATASPLIT_ROOT / self.dataset_name / f"edge-{self.num_splits}splits"

    def edge_split_paths(self, split_i, mkdir=False) -> Dict[str, Path]:
        if mkdir:
            self.edge_split_root.mkdir(parents=True, exist_ok=True)

        split_paths = {}
        for mode in ['train', 'val', 'test']:
            split_paths[f"{mode}_data"] = self.edge_split_root / f"{mode}_data-split{split_i}.pt"
        return split_paths

    def load_edge_split(self, split_i):
        split_paths = self.edge_split_paths(split_i)
        if not all([p.exists() for p in split_paths.values()]):
            self.generate_edge_splits()

        return {key: torch.load(path) for key, path in split_paths.items()}

    def generate_edge_splits(self):
        assert is_undirected(self.data.edge_index)

        for split_i in range(self.num_splits):
            set_seed(seed=split_i * 101)

            transform = RandomLinkSplit(is_undirected=True,
                                        num_val=0.16,
                                        num_test=0.2,
                                        neg_sampling_ratio=1.0,
                                        split_labels=False)
            train_data, val_data, test_data = transform(self.data)

            split_paths = self.edge_split_paths(split_i=split_i, mkdir=True)
            for mode, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
                torch.save(data.detach().clone(), split_paths[f"{mode}_data"])

        logger.info(f"edge splits generated for {self.dataset_name}")


if __name__ == '__main__':
    dataset_names = ["CitationFull-CiteSeer", "CitationFull-Cora_ML", "CitationFull-PubMed", "Amazon-Photo", "GitHub"]
    for dataset_name in dataset_names:
        logger.info(f"Generating/loading data splits for dataset {dataset_name}...")
        data_split = DataSplit(dataset_name, num_splits=5)

        """edge splits"""
        # data_split.generate_edge_splits()
        edge_split = data_split.load_edge_split(split_i=0)
        print("edge_split:", edge_split)

        """node splits"""
        # data_split.generate_node_splits()
        node_split = data_split.load_node_split(split_i=0)
        print("node_split:", node_split)
