# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2023 Nebuly
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import traceback
from functools import partial
from timeit import default_timer as timer
from typing import List, Union, Optional

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv, RGATConv
from tqdm import tqdm

from forward_learning.nodeclass.common import LayerNormalization
from forward_learning.utils import PerformanceManager, load_perf_dict
from gnn import GNNConv
from utils import logger, ResultManager, EarlyStopping


class GNNTop2InputSFLayer(torch.nn.Module):
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

    def forward(self, x_prev, x_same, x_next, edge_index, edge_type=None):
        x = torch.cat((self.norm(x_prev), self.norm(x_next)), dim=1).detach()

        if isinstance(self.gnn_layer.gnn, (RGCNConv, RGATConv)):
            new_x = self.gnn_layer(x, edge_index, edge_type)
        else:
            new_x = self.gnn_layer(x, edge_index)
        return new_x

    def forward_train(
            self,
            x_prev: torch.Tensor,
            x_same: torch.Tensor,
            x_next: torch.Tensor,
            graph_augmenter,
            train_mask,
            edge_index,
            edge_type=None
    ):
        node_emb = self.forward(
            x_prev.detach(),
            x_same.detach(),
            x_next.detach(),
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

        return node_emb, loss.item()


class GNNTop2InputSF(torch.nn.Module):
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
        assert self.append_label in [None, "input"], self.append_label
        self.alternating_update = args.alternating_update
        self.args = args

        self.time_steps = args.epochs
        self.test_time_steps = args.test_time_steps
        self.storable_time_steps = args.storable_time_steps
        self.states = []
        self.layer_sizes = layer_sizes  # [num-inputs, hid-dim-1, hid-dim-2, ..., hid-dim-H, num-classes]
        self.layers = torch.nn.ModuleList()

        for i in range(1, len(layer_sizes) - 1):
            if self.append_label == "input" and i == 1:
                in_channels = num_classes + layer_sizes[i - 1] + layer_sizes[i + 1]
            else:
                in_channels = layer_sizes[i - 1] + layer_sizes[i + 1]

            self.layers.append(
                GNNTop2InputSFLayer(
                    gnn_layer=GNNConv(gnn=gnn,
                                      in_channels=in_channels,
                                      out_channels=layer_sizes[i]),
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

    @torch.no_grad()
    def bottom_up(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            edge_index: torch.Tensor,
            edge_type: Optional[torch.Tensor] = None
    ):
        """initialize entity states (embeddings) via a single bottom-up pass"""
        assert x.ndim == 2 and y.ndim == 2, (x.ndim, y.ndim)

        x_proj = x
        states = [x_proj]

        for layer_i, layer in enumerate(self.layers):
            x_proj = layer.forward(
                x_prev=x_proj,
                x_same=torch.zeros((x.shape[0], layer.gnn_layer.out_channels), device=self.args.device),
                x_next=torch.zeros((x.shape[0], layer.gnn_layer.in_channels - x_proj.shape[1]), device=self.args.device),
                edge_index=edge_index,
                edge_type=edge_type,
            ).detach()
            states.append(x_proj)

        states.append(y)

        return states  # [input x, embeddings from gnn layers, output y]

    # noinspection PyUnusedLocal
    def forward_train(
            self,
            data: torch_geometric.data.Data,
            result_manager: Optional[ResultManager] = None,
            run_i: Optional[int] = None
    ):
        self.train()
        start = timer()
        get_perf_dict = partial(load_perf_dict, start_time=start)

        data = data.to(self.args.device)

        self.graph_augmenter = GraphAugmenter(
            data=data,
            aug_edge_direction=self.args.aug_edge_direction,
            append_label=self.append_label,
            device=self.args.device)
        assert self.graph_augmenter.num_classes == self.num_classes

        aug_graph = self.graph_augmenter.augment(y=data.y, aug_node_mask=data.train_mask)
        self.aug_graph = aug_graph

        states = self.bottom_up(
            x=aug_graph.x,
            y=aug_graph.node_one_hot_labels.float(),  # shape=(num-real-nodes + num-classes, num-classes)
            edge_index=aug_graph.edge_index,
            edge_type=aug_graph.edge_type
        )

        early_stopper = EarlyStopping(self.args.patience) if self.args.patience >= 0 else None
        perf_manager = PerformanceManager("Acc")
        time_step_tqdm = tqdm(range(self.time_steps))
        time_step = -1
        try:
            if self.alternating_update:  # update even layers first, and then update odd layers
                layers_list = [
                    [(j, layer) for j, layer in enumerate(self.layers) if j % 2 == 0],
                    [(j, layer) for j, layer in enumerate(self.layers) if j % 2 == 1]
                ]
            else:  # update all layers simultaneously
                layers_list = [list(enumerate(self.layers))]

            for time_step in range(self.time_steps):
                self.train()

                x = aug_graph.x
                new_states: List[Optional[torch.Tensor]] = [None] * len(states)
                new_states[0] = x

                ts_loss = 0.0
                for layers in layers_list:
                    for j, layer in layers:
                        if self.alternating_update and new_states[j] is not None:
                            x_prev = new_states[j].float()
                        else:
                            x_prev = states[j].float()

                        if self.alternating_update and new_states[j + 2] is not None:
                            x_next = new_states[j + 2].float()
                        else:
                            x_next = states[j + 2].float()

                        node_emb, loss = layer.forward_train(
                            x_prev=x_prev, x_same=states[j + 1], x_next=x_next,
                            graph_augmenter=self.graph_augmenter,
                            train_mask=data.train_mask,
                            edge_index=aug_graph.edge_index,
                            edge_type=aug_graph.edge_type
                        )

                        new_states[j + 1] = node_emb.detach()
                        ts_loss += loss

                new_states[-1] = states[-1]

                assert len(states) == len(new_states), (len(states), len(new_states))
                states = new_states

                """Validation and Testing"""
                if early_stopper is not None and (time_step + 1) % self.args.val_every == 0 \
                        and time_step > self.args.val_from:
                    val_acc, _ = self.eval_model(eval_mask=data.val_mask)
                    if perf_manager.update_val_perf(val_acc, time_step):
                        test_acc, _ = self.eval_model(eval_mask=data.test_mask)
                        perf_manager.update_test_perf(test_acc, time_step)

                    if early_stopper.step(val_acc, self):
                        print(f"[Time-{time_step}] Early stop!")
                        break

                msg = f"[T-{time_step}] TrainLoss: {ts_loss:.4f} | " \
                      f"{' | '.join([perf_manager.val_perf_summary(), perf_manager.test_perf_summary()])}"
                time_step_tqdm.set_description(msg)
                if perf_manager.best_val_epoch == time_step:
                    print(f"\n{msg}\n")

            if early_stopper is not None and early_stopper.best_score is not None:
                early_stopper.load_checkpoint(self)
        except KeyboardInterrupt:
            print(f"\n=== TRAINING INTERRUPTED AT TIME-{time_step}! ===\n")
            if early_stopper is not None and early_stopper.best_score is not None:
                early_stopper.load_checkpoint(self)
        except Exception:
            traceback.print_exc()
            raise

        logger.info("Finished training the network.")
        return get_perf_dict(
            train_epochs=[time_step],
            best_val_epochs=[perf_manager.best_val_epoch]
        )

    @torch.no_grad()
    def eval_model(
            self,
            eval_mask: torch.BoolTensor,
            print_eval=False
    ):
        assert self.aug_graph.real_node_mask.sum() == len(eval_mask)

        self.eval()
        aug_graph = self.aug_graph.to(self.args.device)

        if self.alternating_update:  # update even layers first, and then update odd layers
            layers_list = [
                [(j, layer) for j, layer in enumerate(self.layers) if j % 2 == 0],
                [(j, layer) for j, layer in enumerate(self.layers) if j % 2 == 1]
            ]
        else:
            layers_list = [list(enumerate(self.layers))]

        states = self.bottom_up(
            x=aug_graph.x,
            y=aug_graph.node_one_hot_labels.float(),  # shape=(num-nodes + num-classes, num-classes)
            edge_index=aug_graph.edge_index,
            edge_type=aug_graph.edge_type,
        )

        accumulated_probs = []
        storable_layers = list(range(len(self.layers)))

        for time_step in range(self.test_time_steps):
            x = aug_graph.x
            new_states: List[Optional[torch.Tensor]] = [None] * len(states)
            new_states[0] = x

            for layers in layers_list:
                for j, layer in layers:
                    if self.alternating_update and new_states[j] is not None:
                        x_prev = new_states[j].float()
                    else:
                        x_prev = states[j].float()

                    if self.alternating_update and new_states[j + 2] is not None:
                        x_next = new_states[j + 2].float()
                    else:
                        x_next = states[j + 2].float()

                    x = layer.forward(
                        x_prev=x_prev, x_same=states[j + 1], x_next=x_next,
                        edge_index=aug_graph.edge_index, edge_type=aug_graph.edge_type
                    )
                    new_states[j + 1] = x

                    """compute class prob"""
                    eval_node_feats = x[self.graph_augmenter.real_node_mask][eval_mask]
                    virtual_node_feats = x[~self.graph_augmenter.real_node_mask]

                    logits = torch.mm(eval_node_feats, virtual_node_feats.t())  # shape=(num-eval-nodes, num-classes)
                    if self.args.temperature != 1.0:
                        logits /= self.args.temperature
                    prob = torch.softmax(logits, dim=1)

                    if time_step in self.storable_time_steps and j in storable_layers:
                        accumulated_probs.append(prob.detach())

            new_states[-1] = states[-1]
            states = new_states

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
                (1 * self.num_classes,),
                fill_value=0.0
            ).to(self.device)

        virtual_node_one_hot_labels = torch.cat([
            self.one_hot_labels
        ], dim=1)

        node_one_hot_labels = torch.cat([
            real_node_one_hot_labels,
            virtual_node_one_hot_labels
        ], dim=0)

        if self.append_label is not None:
            aug_node_feats = torch.cat([
                self.aug_x, node_one_hot_labels
            ], dim=1)  # shape=(# total nodes + # classes, # feats + # classes)
        else:
            aug_node_feats = self.aug_x  # shape=(# total nodes + # classes, # feats)

        """augment node mask"""
        aug_node_mask = torch.cat([  # real nodes connected to virtual nodes are set to True
            aug_node_mask,
            torch.zeros(self.num_classes).bool().to(self.device)
        ])

        return Data(x=aug_node_feats, node_one_hot_labels=node_one_hot_labels,
                    y=self.aug_y, edge_index=aug_edge_index, edge_type=aug_edge_type,
                    num_nodes=self.num_nodes + self.num_classes,
                    aug_node_mask=aug_node_mask, real_node_mask=self.real_node_mask)
