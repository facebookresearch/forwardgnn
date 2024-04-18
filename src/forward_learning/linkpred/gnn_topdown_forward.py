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
from typing import List, Optional

import numpy as np
import torch
import torch_geometric
from torch_geometric.nn import RGCNConv, RGATConv
from tqdm import tqdm

from forward_learning.linkpred.common import LayerNormalization
from forward_learning.utils import PerformanceManager, load_perf_dict
from gnn import GNNConv
from utils import logger, eval_link_prediction, ResultManager, EarlyStopping


class GNNTowDownForwardLayer(torch.nn.Module):
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
        self.args = args
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x_prev, x_same, x_next, edge_index, edge_type=None):
        if x_next is not None:
            x = torch.cat((self.norm(x_prev), self.norm(x_next)), dim=1).detach()
        else:
            x = self.norm(x_prev).detach()

        if isinstance(self.gnn_layer.gnn, (RGCNConv, RGATConv)):
            new_x = self.gnn_layer(x, edge_index, edge_type)
        else:
            new_x = self.gnn_layer(x, edge_index)

        return new_x

    @staticmethod
    def link_predict(z, edge_label_index):
        pred = (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=1)
        assert len(pred) == edge_label_index.shape[1], (len(pred), edge_label_index.shape[1])
        return pred

    def forward_train(
        self,
        x_prev: torch.Tensor,
        x_same: torch.Tensor,
        x_next: torch.Tensor,
        train_data: torch_geometric.data.Data,
    ):
        self.optimizer.zero_grad()

        z = self.forward(
            x_prev.detach(),
            x_same.detach(),
            x_next.detach()if x_next is not None else x_next,
            train_data.edge_index,
        )
        out = self.link_predict(z, train_data.edge_label_index)
        loss = self.criterion(out, train_data.edge_label)  # shape=(# pos and neg edges,)

        pos_edge_mask = train_data.edge_label == 1
        neg_edge_mask = train_data.edge_label == 0

        cumulated_logits_pos = loss[pos_edge_mask].exp().mean().item()
        cumulated_logits_neg = (1 - loss[neg_edge_mask].exp()).mean().item()

        loss = loss.mean()
        loss.backward()

        self.optimizer.step()

        logits = [cumulated_logits_pos, cumulated_logits_neg]
        return z, (out[pos_edge_mask], out[neg_edge_mask]), logits

    @torch.no_grad()
    def predict(
            self,
            x_prev: torch.Tensor,
            x_same: torch.Tensor,
            x_next: torch.Tensor,
            edge_index: torch.Tensor,
            edge_label_index: torch.Tensor):
        node_emb = self.forward(x_prev, x_same, x_next, edge_index)
        out = self.link_predict(node_emb, edge_label_index)
        return node_emb, out


class GNNTowDownForward(torch.nn.Module):
    def __init__(
        self,
        layer_sizes: list,
        optimizer_name: str,
        optimizer_kwargs: dict,
        loss_fn_name: str,
        gnn: str,
        args: argparse.Namespace,
    ):
        super().__init__()
        self.theta = args.ff_theta
        self.args = args
        self.alternating_update = args.alternating_update

        self.time_steps = args.epochs
        self.test_time_steps = args.test_time_steps
        self.storable_time_steps = args.storable_time_steps
        self.states = []
        self.layer_sizes = layer_sizes  # [num-inputs, hid-dim-1, hid-dim-2, ..., hid-dim-H, num-classes]
        self.layers = torch.nn.ModuleList()

        for i in range(1, len(layer_sizes) - 1):
            self.layers.append(
                GNNTowDownForwardLayer(
                    gnn_layer=GNNConv(
                        gnn=gnn,
                        in_channels=layer_sizes[i - 1] + layer_sizes[i + 1],
                        out_channels=layer_sizes[i],
                    ),
                    optimizer_name=optimizer_name,
                    optimizer_kwargs=optimizer_kwargs,
                    args=args,
                )
            )

    @torch.no_grad()
    def bottom_up(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_type: Optional[torch.Tensor] = None
    ):
        """initialize entity states (embeddings) via a single bottom-up pass"""
        assert x.ndim == 2, x.ndim

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

        states.append(None)

        return states

    # noinspection PyUnusedLocal
    def forward_train(
            self,
            train_data: torch_geometric.data.Data,
            val_data: torch_geometric.data.Data,
            test_data: torch_geometric.data.Data,
            result_manager: Optional[ResultManager] = None,
            run_i: Optional[int] = None,
    ):
        train_data = train_data.clone().to(self.args.device)
        val_data = val_data.clone().to(self.args.device)
        test_data = test_data.clone().to(self.args.device)

        start = timer()
        get_perf_dict = partial(load_perf_dict, start_time=start)

        states = self.bottom_up(x=train_data.x, edge_index=train_data.edge_index)
        accumulated_goodness = np.array([0.0, 0.0])

        early_stopper = EarlyStopping(self.args.patience) if self.args.patience >= 0 else None
        perf_manager = PerformanceManager("AUC")
        time_step_tqdm = tqdm(range(self.time_steps))
        time_step = -1
        try:
            if self.alternating_update:  # update odd layers first, and then update even layers
                layers_list = [
                    [(j, layer) for j, layer in enumerate(self.layers) if j % 2 == 0],
                    [(j, layer) for j, layer in enumerate(self.layers) if j % 2 == 1]
                ]
            else:  # update all layers simultaneously
                layers_list = [list(enumerate(self.layers))]

            for time_step in range(self.time_steps):
                self.train()

                x = train_data.x
                new_states: List[Optional[torch.Tensor]] = [None] * len(states)
                new_states[0] = x

                ts_goodnesses = np.array([0.0, 0.0])  # [goodness of positives, goodness of negatives]
                for layers in layers_list:
                    for j, layer in layers:
                        if self.alternating_update and new_states[j] is not None:
                            x_prev = new_states[j].float()
                        else:
                            x_prev = states[j].float()

                        if self.alternating_update and new_states[j + 2] is not None:
                            x_next = new_states[j + 2].float() if new_states[j + 2] is not None else None
                        else:
                            x_next = states[j + 2].float() if states[j + 2] is not None else None

                        node_emb, out, goodnesses = layer.forward_train(
                            x_prev=x_prev, x_same=states[j + 1], x_next=x_next, train_data=train_data,
                        )

                        new_states[j + 1] = node_emb.detach()
                        accumulated_goodness += goodnesses
                        ts_goodnesses += goodnesses

                new_states[-1] = states[-1]

                assert len(states) == len(new_states), (len(states), len(new_states))
                states = new_states

                """Validation and Testing"""
                if early_stopper is not None and (time_step + 1) % self.args.val_every == 0 \
                        and time_step > self.args.val_from:
                    val_auc, _ = self.eval_model(eval_data=val_data)
                    if perf_manager.update_val_perf(val_auc, time_step):
                        test_auc, _ = self.eval_model(eval_data=test_data)
                        perf_manager.update_test_perf(test_auc, time_step)

                    if early_stopper.step(val_auc, self):
                        print(f"[Time-{time_step}] Early stop!")
                        break

                msg = f"[T-{time_step}] Pos={ts_goodnesses[0]:.4f}, Neg={ts_goodnesses[1]:.4f} | " \
                      f"TrainSep: {ts_goodnesses[0] - ts_goodnesses[1]:.2f} | " \
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
            eval_data: torch_geometric.data.Data,
            print_eval=False
    ):
        self.eval()

        eval_data = eval_data.to(self.args.device)

        if self.alternating_update:  # update even layers first, and then update odd layers
            layers_list = [
                [(j, layer) for j, layer in enumerate(self.layers) if j % 2 == 0],
                [(j, layer) for j, layer in enumerate(self.layers) if j % 2 == 1]
            ]
        else:
            layers_list = [list(enumerate(self.layers))]

        accumulated_goodness_list = []
        states = self.bottom_up(
            x=eval_data.x,
            edge_index=eval_data.edge_index,
        )

        storable_layers = list(range(len(self.layers)))
        for time_step in range(self.test_time_steps):
            x = eval_data.x
            new_states: List[Optional[torch.Tensor]] = [None] * len(states)
            new_states[0] = x

            for layers in layers_list:
                for j, layer in layers:
                    if self.alternating_update and new_states[j] is not None:
                        x_prev = new_states[j].float()
                    else:
                        x_prev = states[j].float()

                    if self.alternating_update and new_states[j + 2] is not None:
                        x_next = new_states[j + 2].float() if new_states[j + 2] is not None else new_states[j + 2]
                    else:
                        x_next = states[j + 2].float() if states[j + 2] is not None else states[j + 2]

                    x, out = layer.predict(
                        x_prev=x_prev, x_same=states[j + 1], x_next=x_next,
                        edge_index=eval_data.edge_index, edge_label_index=eval_data.edge_label_index,
                    )
                    new_states[j + 1] = x

                    if time_step in self.storable_time_steps and j in storable_layers:
                        accumulated_goodness_list.append(out.sigmoid())

            new_states[-1] = states[-1]
            states = new_states

        accumulated_goodness = torch.stack(accumulated_goodness_list).mean(dim=0)

        eval_perf_dict = eval_link_prediction(
            y_true=eval_data.edge_label,
            y_score=accumulated_goodness,
        )
        rocauc = eval_perf_dict['rocauc']

        if print_eval:
            print("\nTest ROCAUC: {:.6f}".format(rocauc))

        return rocauc, accumulated_goodness
