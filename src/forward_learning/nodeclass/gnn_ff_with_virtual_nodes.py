# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import traceback
from functools import partial
from timeit import default_timer as timer
from typing import List, Union, Optional, Dict

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv, RGATConv
from tqdm import tqdm

from forward_learning.loss import forwardforward_loss_fn
from forward_learning.nodeclass.common import LayerNormalization
from forward_learning.utils import PerformanceManager, load_perf_dict
from gnn import GNNConv, CachedSAGEConv, CachedGCNConv
from utils import logger, ResultManager, EarlyStopping


class GNNVirtualNodeFFLayer(torch.nn.Module):
    def __init__(
        self,
        gnn_layer,
        optimizer_name: str,
        optimizer_kwargs: dict,
        loss_fn_name: str,
        args: argparse.Namespace,
    ):
        super().__init__()
        self.gnn_layer = gnn_layer
        self.optimizer = getattr(torch.optim, optimizer_name)(
            gnn_layer.parameters(), **optimizer_kwargs
        )
        self.loss_fn_name = loss_fn_name
        self.args = args

    def forward(self, x, edge_index, edge_type=None):
        if isinstance(self.gnn_layer.gnn, (RGCNConv, RGATConv)):
            return self.gnn_layer(x, edge_index, edge_type)
        else:
            return self.gnn_layer(x, edge_index)

    def clear_cached_propagate(self):
        if isinstance(self.gnn_layer.gnn, (CachedSAGEConv, CachedGCNConv)):
            self.gnn_layer.gnn.clear_cached()

    def forward_train(
        self,
        pos_graph: torch_geometric.data.Data,
        neg_graphs: List[torch_geometric.data.Data],
        theta: float,
    ):
        pos_target = torch.ones(pos_graph.aug_node_mask.sum(), device=pos_graph.x.device)
        neg_target = torch.zeros(neg_graphs[0].aug_node_mask.sum(), device=pos_graph.x.device)

        y_pos = self.forward(pos_graph.x, pos_graph.edge_index, pos_graph.edge_type)

        y_neg_list = []
        for neg_graph in neg_graphs:
            y_neg = self.forward(neg_graph.x, neg_graph.edge_index, neg_graph.edge_type)
            y_neg_list.append(y_neg)

        if self.loss_fn_name == "forwardforward_loss_fn":
            loss_pos, cumulated_logits_pos = forwardforward_loss_fn(y_pos[pos_graph.aug_node_mask], theta, pos_target)
        else:
            raise ValueError(f"Undefined: {self.loss_fn_name}")

        cumulated_logits_neg_sum, num_total_negs = 0, 0
        loss_neg_parts = []
        for neg_graph, y_neg in zip(neg_graphs, y_neg_list):
            if self.loss_fn_name == "forwardforward_loss_fn":
                loss_neg_part, cumulated_logits_neg_part = \
                    forwardforward_loss_fn(y_neg[neg_graph.aug_node_mask], theta, neg_target)
            else:
                raise ValueError(f"Undefined: {self.loss_fn_name}")

            num_negs = neg_graph.aug_node_mask.sum().item()
            cumulated_logits_neg_sum += cumulated_logits_neg_part * num_negs
            num_total_negs += num_negs
            loss_neg_parts.append(loss_neg_part)

        loss_neg = torch.mean(torch.stack(loss_neg_parts))
        cumulated_logits_neg = cumulated_logits_neg_sum / num_total_negs

        self.optimizer.zero_grad()
        loss = loss_pos + loss_neg
        loss.backward()
        self.optimizer.step()

        logits = [cumulated_logits_pos, cumulated_logits_neg]
        return (y_pos, y_neg_list), logits

    @torch.no_grad()
    def forward_predict(
            self,
            input_feats: torch.Tensor,
            theta: float,
            edge_index: torch.Tensor,
            edge_type: torch.Tensor
    ):
        """Evaluate the layer with the given input and theta."""
        if isinstance(self.gnn_layer.gnn, (RGCNConv, RGATConv)):
            y = self.forward(input_feats, edge_index, edge_type)
        else:
            y = self.forward(input_feats, edge_index)
        return y, torch.square(y).mean(dim=1) - theta

    @property
    def requires_training(self):
        return True


class GNNVirtualNodeLabelAppendingFF(torch.nn.Module):
    def __init__(
        self,
        layer_sizes: list,
        optimizer_name: str,
        optimizer_kwargs: dict,
        loss_fn_name: str,
        num_classes: int,
        gnn: str,
        append_label: Union[str, None],
        args: argparse.Namespace,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.theta = args.ff_theta
        self.patience = args.patience
        self.val_every = args.val_every
        self.append_label = append_label
        self.args = args

        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(LayerNormalization())

            if self.append_label == "all":
                in_channels = layer_sizes[i] + num_classes
            elif self.append_label == "input":
                if i == 0:
                    in_channels = layer_sizes[i] + num_classes
                else:
                    in_channels = layer_sizes[i]
            else:  # don't append label to node features
                in_channels = layer_sizes[i]

            self.layers.append(
                GNNVirtualNodeFFLayer(
                    gnn_layer=GNNConv(gnn=gnn,
                                      in_channels=in_channels,
                                      out_channels=layer_sizes[i + 1]),
                    optimizer_name=optimizer_name,
                    optimizer_kwargs=optimizer_kwargs,
                    loss_fn_name=loss_fn_name,
                    args=args,
                )
            )

        self.graph_augmenter: GraphAugmenter = None
        self.eval_aug_graph_dict: Optional[List[torch_geometric.data.Data]] = None

    def forward(self, x, edge_index, edge_type):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (RGCNConv, RGATConv)):
                x = layer(x, edge_index, edge_type)
            else:
                x = layer(x, edge_index)
        return x

    def create_pos_and_neg_graphs(
            self,
            num_negs: int,
            aug_node_mask: torch.Tensor
    ):
        num_negs = min(num_negs, self.graph_augmenter.num_classes - 1)
        pos_graph = self.graph_augmenter.augment(y=self.graph_augmenter.y, aug_node_mask=aug_node_mask)

        # for each node, randomly select its negative classes
        x = torch.arange(self.graph_augmenter.num_classes).repeat(self.graph_augmenter.num_nodes, 1)  # shape=(num nodes, num classes)
        # ref: https://discuss.pytorch.org/t/what-is-the-most-efficient-way-to-shuffle-each-row-of-a-tensor-with-different-shuffling-order-for-each-of-the-row/109772
        r = torch.rand_like(x.float())
        r[torch.arange(self.graph_augmenter.num_nodes), self.graph_augmenter.y] = 10000000  # to exclude the true class of each node
        assert self.graph_augmenter.num_classes < 10000000

        indices = torch.argsort(r, dim=-1, descending=False)
        neg_classes = torch.gather(x, dim=-1, index=indices)[:, 0:-1]  # shape=(num nodes, num classes - 1). 0:-1 to exclude the true class
        neg_graphs = []
        for i in range(min(num_negs, neg_classes.shape[1])):
            neg_graph = self.graph_augmenter.augment(
                y=neg_classes[:, i:i + 1].view(-1),
                aug_node_mask=aug_node_mask,
                node_feat_y=self.graph_augmenter.y
            )
            neg_graphs.append(neg_graph)

        return pos_graph, neg_graphs

    def create_eval_aug_graphs(self, node_labels, train_mask, eval_mask) -> Dict[int, torch_geometric.data.Data]:
        eval_aug_graph_dict = {}

        for class_label in range(self.num_classes):
            node_labels_for_class = torch.full_like(node_labels, fill_value=-1, device=node_labels.device)
            node_labels_for_class[train_mask] = node_labels[train_mask]
            node_labels_for_class[eval_mask] = class_label

            aug_graph = self.graph_augmenter.augment(y=node_labels_for_class, aug_node_mask=train_mask | eval_mask)
            assert aug_graph.num_nodes == self.graph_augmenter.num_nodes + self.num_classes
            eval_aug_graph_dict[class_label] = aug_graph

        return eval_aug_graph_dict

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

        pos_graph, neg_graphs = self.create_pos_and_neg_graphs(
            num_negs=self.args.num_negs,
            aug_node_mask=data.train_mask
        )

        val_aug_graphs = self.create_eval_aug_graphs(
            node_labels=data.y,
            train_mask=data.train_mask,
            eval_mask=data.val_mask
        )
        test_aug_graphs = self.create_eval_aug_graphs(
            node_labels=data.y,
            train_mask=data.train_mask | data.val_mask,
            eval_mask=data.test_mask
        )

        train_epochs, best_val_epochs = [], []
        get_perf_dict = partial(load_perf_dict, start, train_epochs, best_val_epochs)

        for i, layer in enumerate(self.layers):
            if layer.requires_training:
                early_stopper = EarlyStopping(self.patience) if self.patience >= 0 else None
                perf_manager = PerformanceManager("Acc")

                epoch_tqdm, epoch = tqdm(range(self.args.epochs)), -1
                try:
                    layer.clear_cached_propagate()

                    for epoch in epoch_tqdm:
                        self.train()

                        _, logits = layer.forward_train(
                            pos_graph,
                            neg_graphs,
                            self.theta
                        )

                        """Validation and Testing"""
                        if early_stopper is not None and (epoch + 1) % self.val_every == 0 and epoch > self.args.val_from:
                            val_acc, _ = self.eval_model(data, train_mask=data.train_mask, eval_mask=data.val_mask,
                                                         eval_aug_graphs=val_aug_graphs,
                                                         last_eval_layer=i, print_eval=False)
                            if perf_manager.update_val_perf(val_acc, epoch):
                                test_acc, _ = self.eval_model(data, train_mask=data.train_mask | data.val_mask,
                                                              eval_mask=data.test_mask,
                                                              eval_aug_graphs=test_aug_graphs,
                                                              last_eval_layer=i, print_eval=False)
                                perf_manager.update_test_perf(test_acc, epoch)

                            if early_stopper.step(val_acc, layer):
                                print(f"[Layer {i}: Epoch-{epoch}] Early stop!")
                                break

                        epoch_tqdm.set_description(
                            f"[Layer {i}: Epoch-{epoch}] Pos={logits[0]:.4f}, Neg={logits[1]:.4f} | "
                            f"TrainSep: {logits[0] - logits[1]:.4f} | "
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

                test_acc, _ = self.eval_model(data, train_mask=data.train_mask | data.val_mask,
                                              eval_mask=data.test_mask,
                                              eval_aug_graphs=test_aug_graphs,
                                              last_eval_layer=i, print_eval=False)
                print(f"[Layer-{i}] Test Accuracy : {test_acc:.2f}%\n")
                assert (i + 1) % 2 == 0, i
                result_manager.save_run_result(run_i, perf_dict=get_perf_dict(perf=test_acc), num_layers=(i + 1) // 2)  # save as layer 1, 2, 3, ...

                node_feats = layer.forward(pos_graph.x, pos_graph.edge_index, pos_graph.edge_type)
                if self.append_label == "all":
                    aug_node_feats = torch.cat([
                        node_feats, pos_graph.node_one_hot_labels,
                    ], dim=1)
                else:
                    aug_node_feats = node_feats

                node_feats = node_feats.detach()
                aug_node_feats = aug_node_feats.detach()

                # noinspection PyPropertyAccess
                pos_graph.x = aug_node_feats
                for neg_graph in neg_graphs:  # noinspection PyPropertyAccess
                    neg_graph.x = aug_node_feats  # neg_graph uses the same features as pos_graph, yet has different links to virtual nodes
            else:  # normalization layer
                # noinspection PyPropertyAccess
                pos_graph.x = layer(pos_graph.x).detach()
                for neg_graph in neg_graphs:  # noinspection PyPropertyAccess
                    neg_graph.x = layer(neg_graph.x).detach()

        logger.info("Finished training the network.")
        return get_perf_dict()

    @torch.no_grad()
    def eval_model(
            self,
            data: torch_geometric.data.Data,
            *,
            train_mask: torch.BoolTensor,
            eval_mask: torch.BoolTensor,
            eval_aug_graphs=None,
            last_eval_layer: int = -1,
            print_eval=True
    ):
        self.eval()
        data = data.to(self.args.device)
        input_node_feats, node_labels, edge_index = data.x, data.y, data.edge_index

        accumulated_goodness_list = []
        if last_eval_layer < 0:
            last_eval_layer = len(self.layers) - 1

        if eval_aug_graphs is None:
            eval_aug_graphs = self.create_eval_aug_graphs(node_labels, train_mask, eval_mask)

        accumulate_from = 0 if last_eval_layer >= 2 else 0

        for class_label in range(self.num_classes):
            aug_graph = eval_aug_graphs[class_label]

            node_feats = aug_graph.x
            accumulated_goodness = torch.zeros(data.num_nodes, device=self.args.device)
            for i, layer in enumerate(self.layers):
                if i <= last_eval_layer:
                    node_feats, goodness = \
                        layer.forward_predict(node_feats, self.theta, aug_graph.edge_index, aug_graph.edge_type)

                    if layer.requires_training and self.append_label == "all":
                        node_feats = torch.cat([
                            node_feats, aug_graph.node_one_hot_labels
                        ], dim=1)

                    if i >= accumulate_from:
                        accumulated_goodness += goodness[aug_graph.real_node_mask]  # exclude virtual nodes.

            accumulated_goodness_list.append(accumulated_goodness.view(-1, 1))

        all_goodness = torch.cat(accumulated_goodness_list, dim=1)  # shape=(# nodes, # classes)
        pred = all_goodness[eval_mask].argmax(dim=1)
        target = data.y[eval_mask]
        # noinspection PyUnresolvedReferences
        correct = (pred == target).int().sum().item()
        acc = 100.0 * correct / target.shape[0]
        if print_eval:
            print("\nTest Accuracy: {:.4f}% ({}/{})".format(acc, correct, target.shape[0]))

        return acc, accumulated_goodness_list


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
