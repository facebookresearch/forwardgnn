# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import traceback
from functools import partial
from timeit import default_timer as timer
from typing import Union, Optional

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv, RGATConv
from tqdm import tqdm

from forward_learning.nodeclass.common import LayerNormalization
from forward_learning.utils import PerformanceManager, load_perf_dict
from gnn import GNNConv, CachedSAGEConv, CachedGCNConv
from utils import logger, ResultManager, EarlyStopping


class GNNSingleForwardLayer(torch.nn.Module):
    def __init__(
            self,
            gnn_layer,
            optimizer_name: str,
            optimizer_kwargs: dict,
            args: argparse.Namespace,
    ):
        super().__init__()
        self.norm = LayerNormalization()
        self.gnn_layer = gnn_layer
        self.optimizer = getattr(torch.optim, optimizer_name)(
            gnn_layer.parameters(), **optimizer_kwargs
        )
        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        self.temperature = args.temperature

    def forward(self, x, edge_index, edge_type=None):
        x = self.norm(x)

        if isinstance(self.gnn_layer.gnn, (RGCNConv, RGATConv)):
            return self.gnn_layer(x, edge_index, edge_type)
        else:
            return self.gnn_layer(x, edge_index)

    def clear_cached_propagate(self):
        if isinstance(self.gnn_layer.gnn, (CachedSAGEConv, CachedGCNConv)):
            self.gnn_layer.gnn.clear_cached()

    def forward_train(self, x, graph_augmenter, train_mask, edge_index, edge_type=None):
        node_emb = self.forward(
            x.detach(),
            edge_index,
            edge_type
        )

        train_node_emb = node_emb[graph_augmenter.real_node_mask][train_mask]
        virtual_node_emb = node_emb[~graph_augmenter.real_node_mask]

        logits = torch.mm(train_node_emb, virtual_node_emb.t())  # shape=(num-train-nodes, num-classes)
        if self.temperature != 1.0:
            logits /= self.temperature
        train_y = graph_augmenter.y[train_mask]

        self.optimizer.zero_grad()
        loss = self.criterion(logits, train_y)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_max_norm)
        self.optimizer.step()

        return loss.item()


class GNNSingleForward(torch.nn.Module):
    def __init__(
            self,
            layer_sizes: list,
            optimizer_name: str,
            optimizer_kwargs: dict,
            num_classes: int,
            gnn: str,
            args: argparse.Namespace,
    ):
        super().__init__()
        self.gnn = gnn
        self.num_classes = num_classes
        self.patience = args.patience
        self.val_every = args.val_every
        self.append_label = args.append_label
        self.args = args

        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            if self.append_label == "input":
                if i == 0:
                    in_channels = layer_sizes[i] + num_classes
                else:
                    in_channels = layer_sizes[i]
            elif self.append_label == "all":
                in_channels = layer_sizes[i] + num_classes
            else:  # don't append one-hot label to node features
                in_channels = layer_sizes[i]

            self.layers.append(
                GNNSingleForwardLayer(
                    gnn_layer=GNNConv(gnn=gnn,
                                      in_channels=in_channels,
                                      out_channels=layer_sizes[i + 1]),
                    optimizer_name=optimizer_name,
                    optimizer_kwargs=optimizer_kwargs,
                    args=args,
                )
            )

        self.graph_augmenter: GraphAugmenter = None
        self.aug_graph: torch_geometric.data.Data = None

    def forward(self, x, edge_index, edge_type=None):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (RGCNConv, RGATConv)):
                x = layer(x, edge_index, edge_type)
            else:
                x = layer(x, edge_index)
        return x

    def forward_train(
            self,
            data: torch_geometric.data.Data,
            result_manager: ResultManager,
            run_i: int
    ):
        self.train()
        start = timer()

        data = data.to(self.args.device)

        self.graph_augmenter = GraphAugmenter(
            data=data,
            aug_edge_direction=self.args.aug_edge_direction,
            append_label=self.append_label,
            device=self.args.device
        )
        assert self.graph_augmenter.num_classes == self.num_classes

        aug_graph = self.graph_augmenter.augment(y=data.y, aug_node_mask=data.train_mask)
        self.aug_graph = aug_graph
        node_feats = aug_graph.x

        train_epochs, best_val_epochs = [], []
        get_perf_dict = partial(load_perf_dict, start, train_epochs, best_val_epochs)

        for i, layer in enumerate(self.layers):
            early_stopper = EarlyStopping(self.patience) if self.patience >= 0 else None
            perf_manager = PerformanceManager("Acc")
            epoch_tqdm, epoch = tqdm(range(self.args.epochs)), -1
            try:
                layer.clear_cached_propagate()

                for epoch in epoch_tqdm:
                    self.train()

                    loss = layer.forward_train(
                        node_feats,
                        self.graph_augmenter,
                        data.train_mask,
                        aug_graph.edge_index,
                        aug_graph.edge_type
                    )

                    """Validation and Testing"""
                    if early_stopper is not None and (epoch + 1) % self.val_every == 0 \
                            and epoch > self.args.val_from:
                        val_acc, _ = self.eval_model(eval_mask=data.val_mask, last_eval_layer=i)
                        if perf_manager.update_val_perf(val_acc, epoch):
                            test_acc, _ = self.eval_model(eval_mask=data.test_mask, last_eval_layer=i)
                            perf_manager.update_test_perf(test_acc, epoch)

                        if early_stopper.step(val_acc, layer):
                            print(f"[Layer {i}: Epoch-{epoch}] Early stop!")
                            break

                    epoch_tqdm.set_description(
                        f"[Layer {i}: Epoch-{epoch}] Loss={loss:.4f} | "
                        f"{' | '.join([perf_manager.val_perf_summary(), perf_manager.test_perf_summary()])}"
                    )

                if early_stopper is not None and early_stopper.best_score is not None:
                    early_stopper.load_checkpoint(layer)
                logger.info(f"Finished training layer {i + 1} / {len(self.layers)}.\n")
            except KeyboardInterrupt:
                print(f"\n=== LAYER-{i} TRAINING INTERRUPTED AT EPOCH-{epoch}! ===\n")
                if early_stopper is not None and early_stopper.best_score is not None:
                    early_stopper.load_checkpoint(layer)
            except Exception:
                traceback.print_exc()
                raise

            train_epochs.append(epoch)
            best_val_epochs.append(perf_manager.best_val_epoch)

            test_acc, _ = self.eval_model(eval_mask=data.test_mask, last_eval_layer=i)
            print(f"[Layer-{i}] Test Accuracy : {test_acc:.2f}%\n")
            result_manager.save_run_result(run_i, perf_dict=get_perf_dict(perf=test_acc), num_layers=i + 1)

            # use output from this gnn layer as the input to the next layer
            node_feats = layer.forward(node_feats, aug_graph.edge_index, aug_graph.edge_type)
            if self.append_label == "all":
                node_feats = torch.cat([node_feats, aug_graph.node_one_hot_labels], dim=1)
            node_feats = node_feats.detach()

        logger.info("Finished training the network.")
        return get_perf_dict()

    @torch.no_grad()
    def eval_model(
            self,
            eval_mask: torch.BoolTensor,
            last_eval_layer: int = -1,
            print_eval=False
    ):
        assert self.aug_graph.real_node_mask.sum() == len(eval_mask)
        self.eval()
        aug_graph = self.aug_graph.to(self.args.device)

        if last_eval_layer < 0:
            last_eval_layer = len(self.layers) - 1

        accumulate_from = 0 if last_eval_layer >= 1 else 0
        node_feats = aug_graph.x
        accumulated_probs = []
        for i, layer in enumerate(self.layers):
            if i <= last_eval_layer:
                node_feats = layer.forward(node_feats, aug_graph.edge_index, aug_graph.edge_type)

                eval_node_feats = node_feats[self.graph_augmenter.real_node_mask][eval_mask]
                virtual_node_feats = node_feats[~self.graph_augmenter.real_node_mask]

                logits = torch.mm(eval_node_feats, virtual_node_feats.t())  # shape=(num-eval-nodes, num-classes)
                if self.args.temperature != 1.0:
                    logits /= self.args.temperature
                prob = torch.softmax(logits, dim=1)

                if i >= accumulate_from:
                    accumulated_probs.append(prob.detach())

                if self.append_label == "all":
                    node_feats = torch.cat([node_feats, aug_graph.node_one_hot_labels], dim=1)

        pred = sum(accumulated_probs).argmax(dim=1)
        target = self.graph_augmenter.y[eval_mask]
        # noinspection PyUnresolvedReferences
        correct = (pred == target).int().sum().item()
        acc = 100.0 * correct / target.shape[0]
        if print_eval:
            print("\nTest Accuracy: {:.4f}% ({}/{})".format(acc, correct, target.shape[0]))

        return acc, accumulated_probs


class GraphAugmenter:
    """Augment the graph with virtual nodes"""
    def __init__(
            self,
            data: torch_geometric.data.Data,
            append_label: Union[str, None],
            aug_edge_direction: str,
            device: torch.device,
            zero_or_uniform_for_non_augmented_nodes: str = 'uniform'
    ):
        x, y, edge_index = data.x, data.y, data.edge_index
        assert y.min() == 0, y.min()
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.device = device

        self.num_classes = y.max() - y.min() + 1
        assert self.num_classes == len(torch.unique(y)), (self.num_classes, len(torch.unique(y)))
        self.num_nodes = data.num_nodes

        self.virtual_node_ids = torch.arange(self.num_nodes, self.num_nodes + self.num_classes).to(device)
        self.virtual_node_feats = nn.Embedding(self.num_classes, x.shape[1]).weight.clone().detach().to(device)

        self.aug_x = torch.cat([
            x,
            self.virtual_node_feats
        ], dim=0)  # assign random features to virtual nodes

        self.aug_y = torch.cat([
            y,
            torch.tensor([-1] * self.num_classes).to(device)
        ], dim=0)  # assign -1 as virtual nodes' labels

        self.real_node_mask = torch.cat([  # real nodes are set to True
            torch.ones(self.num_nodes),
            torch.zeros(self.num_classes)
        ]).bool().to(device)

        self.aug_edge_direction = aug_edge_direction
        assert aug_edge_direction in ['unidirection', 'bidirection'], aug_edge_direction

        self.append_label = append_label  # whether to augment node features with one-hot node label
        self.one_hot_labels = torch.eye(self.num_classes).to(device)  # shape=(num classes, num classes)
        self.zero_or_uniform_for_non_augmented_nodes = zero_or_uniform_for_non_augmented_nodes
        assert zero_or_uniform_for_non_augmented_nodes in ['zero', 'uniform'], zero_or_uniform_for_non_augmented_nodes

    def augment(
            self,
            y: Optional[torch.Tensor] = None,
            aug_node_mask: Optional[torch.Tensor] = None,
            node_feat_y: Optional[torch.Tensor] = None
    ):
        if y is None:
            y = self.y
        assert y.shape == self.y.shape, (y.shape, self.y.shape)
        y = y.to(self.device)
        if node_feat_y is not None:
            node_feat_y = node_feat_y.to(self.device)

        if aug_node_mask is None:  # augment all nodes, if none given
            aug_node_mask = torch.ones(self.num_nodes).bool()
        assert len(aug_node_mask) == len(self.x)
        aug_node_mask = aug_node_mask.to(self.device)

        """augment edge list"""
        aug_nodes = torch.arange(self.num_nodes, device=self.device)[aug_node_mask]
        v_nodes = self.virtual_node_ids[y[aug_node_mask]]

        if self.aug_edge_direction == "bidirection":  # bidirectional: edges from real nodes to virtual nodes & edges from virtual nodes to real nodes
            v_src = torch.cat([aug_nodes, v_nodes])
            v_dst = torch.cat([v_nodes, aug_nodes])
        else:  # unidirectional: edges from real nodes to virtual nodes
            v_src = torch.cat([aug_nodes])
            v_dst = torch.cat([v_nodes])

        v_edge_index = torch.stack([v_src, v_dst])  # shape=(2, # edges to/from virtual nodes)
        aug_edge_index = torch.cat([self.edge_index, v_edge_index], dim=1)

        """augment edge type"""
        if self.aug_edge_direction == "bidirection":
            v_edge_types = torch.arange(1, 1 + 2 * self.num_classes).to(self.device)  # edge type of 0 is for the real edges
            v_edge_type = torch.cat([
                v_edge_types[y[aug_node_mask]],  # edges from real node to virtual node: edge type is between 1 and num-classes
                v_edge_types[y[aug_node_mask] + self.num_classes],  # edges from virtual node to real node: edge type is between 1 + num-classes and 1 + 2 x num-classes
            ])
        else:
            v_edge_types = torch.arange(1, 1 + 1 * self.num_classes).to(self.device)  # edge type of 0 is for the real edges
            v_edge_type = torch.cat([
                v_edge_types[y[aug_node_mask]],  # edges from real node to virtual node: edge type is between 1 and num-classes
            ])
        assert torch.all(v_edge_type >= 1)
        assert len(v_edge_type) == v_edge_index.shape[1]

        aug_edge_type = torch.cat([
            torch.zeros(self.edge_index.shape[1], dtype=torch.long, device=self.device),  # edge type of 0 is for the real edges
            v_edge_type
        ])

        """augment node features"""
        if self.append_label is not None:
            real_node_one_hot_labels = torch.zeros(
                self.num_nodes,
                1 * self.num_classes,
            ).to(self.device)

            if node_feat_y is None:
                node_feat_y = y

            real_node_one_hot_labels[aug_node_mask] = torch.cat([
                self.one_hot_labels[node_feat_y[aug_node_mask]],
            ], dim=1)

            if self.zero_or_uniform_for_non_augmented_nodes == 'uniform':
                # use uniform distribution for non-training nodes
                real_node_one_hot_labels[~aug_node_mask] = torch.cat([
                    torch.full((self.num_classes,), fill_value=1 / self.num_classes).to(self.device),
                ]).repeat((~aug_node_mask).sum(), 1)
            else:
                # use all zeros for non-training nodes
                real_node_one_hot_labels[~aug_node_mask] = torch.full(
                    (1 * self.num_classes,), fill_value=0.0
                ).to(self.device)

            virtual_node_one_hot_labels = torch.cat([
                self.one_hot_labels
            ], dim=1)

            node_one_hot_labels = torch.cat([
                real_node_one_hot_labels,
                virtual_node_one_hot_labels
            ], dim=0)

            aug_node_feats = torch.cat([
                self.aug_x, node_one_hot_labels
            ], dim=1)  # shape=(# total nodes + # classes, # classes)
        else:
            aug_node_feats = self.aug_x
            node_one_hot_labels = None

        """augment node mask"""
        aug_node_mask = torch.cat([  # real nodes connected to virtual nodes are set to True
            aug_node_mask,
            torch.zeros(self.num_classes).bool().to(self.device)
        ])

        return Data(x=aug_node_feats, node_one_hot_labels=node_one_hot_labels,
                    y=self.aug_y, edge_index=aug_edge_index, edge_type=aug_edge_type,
                    num_nodes=self.num_nodes + self.num_classes,
                    aug_node_mask=aug_node_mask, real_node_mask=self.real_node_mask)
