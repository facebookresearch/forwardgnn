# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch_geometric.datasets import CitationFull, Amazon, GitHub
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import is_undirected

import settings
from datasets.datasplit import DataSplit
from utils import logger


def load_node_classification_data(args, split_i):
    dataset = load_dataset(args.dataset)
    data = dataset[0]

    data_split = DataSplit(dataset_name=args.dataset, num_splits=5)
    node_split = data_split.load_node_split(split_i=split_i)
    logger.info(f"loaded node split from {data_split.node_split_root.resolve()}")

    data.train_mask = torch.zeros(data.num_nodes).bool()
    data.train_mask[node_split['train_node_index']] = True
    data.val_mask = torch.zeros(data.num_nodes).bool()
    data.val_mask[node_split['val_node_index']] = True
    data.test_mask = torch.zeros(data.num_nodes).bool()
    data.test_mask[node_split['test_node_index']] = True

    if not hasattr(data, "num_classes") or data.num_classes is None:
        data.num_classes = dataset.num_classes
    if not hasattr(data, "num_features") or data.num_features is None:
        data.num_features = dataset.num_features

    print()
    print(f'Dataset ({args.dataset}):')
    print('================================================================================')
    print(data)
    print(f"- Number of classes: {data.num_classes}")
    print(f"- Number of training nodes: {data.train_mask.sum()}")
    print(f"- Number of validation nodes: {data.val_mask.sum()}")
    print(f"- Number of testing nodes: {data.test_mask.sum()}")
    print('================================================================================')

    return data


def load_link_prediction_data(args, split_i):
    dataset = load_dataset(args.dataset)
    data = dataset[0]

    if not hasattr(data, "num_classes") or data.num_classes is None:
        data.num_classes = dataset.num_classes
    if not hasattr(data, "num_features") or data.num_features is None:
        data.num_features = dataset.num_features

    data_split = DataSplit(dataset_name=args.dataset, num_splits=5)
    edge_split = data_split.load_edge_split(split_i=split_i)
    logger.info(f"loaded edge split from {data_split.edge_split_root.resolve()}")
    train_data, val_data, test_data = edge_split['train_data'], edge_split['val_data'], edge_split['test_data']

    print()
    print(f'Dataset ({args.dataset}):')
    print('================================================================================')
    print("Raw Data:", data)
    print("Train Data:", train_data)
    print("Validation Data:", val_data)
    print("Testing Data:", test_data)
    print('================================================================================')

    return train_data, val_data, test_data, data


def load_dataset(dataset):
    if dataset.startswith("CitationFull"):
        dataset = CitationFull(
            root=f'{settings.DATA_ROOT}/CitationFull',
            name=dataset.strip().split("-")[1],
            transform=NormalizeFeatures()
        )
    elif dataset.startswith("Amazon"):
        dataset = Amazon(
            root=f'{settings.DATA_ROOT}/Amazon',
            name=dataset.strip().split("-")[1],
            transform=NormalizeFeatures()
        )
    elif dataset == "GitHub":
        dataset = GitHub(
            root=f'{settings.DATA_ROOT}/GitHub',
            transform=NormalizeFeatures()
        )
    else:
        raise ValueError(f"Unavailable dataset: {dataset}")

    assert len(dataset) == 1, len(dataset)
    assert is_undirected(dataset[0].edge_index)

    return dataset
