# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import traceback
from functools import partial
from timeit import default_timer as timer

import torch
import torch_geometric
from torch_geometric.data import Data
from tqdm import tqdm

from forward_learning.linkpred.common import BaseForwardLayer, LayerNormalization
from forward_learning.loss import forwardforward_loss_fn
from forward_learning.utils import PerformanceManager, load_perf_dict
from gnn import GNNConv
from utils import logger, eval_link_prediction, ResultManager, EarlyStopping


class GNNForwardLayer(BaseForwardLayer):
    def __init__(
            self,
            gnn_layer: GNNConv,
            optimizer_name: str,
            optimizer_kwargs: dict,
            forward_learning_type: str,
            args: argparse.Namespace,
    ):
        super().__init__()
        self.gnn_layer = gnn_layer
        self.optimizer = getattr(torch.optim, optimizer_name)(
            gnn_layer.parameters(), **optimizer_kwargs
        )
        self.forward_learning_type = forward_learning_type
        self.criterion_no_reduction = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.args = args

    def forward(self, x, edge_index):
        return self.gnn_layer(x, edge_index)

    @staticmethod
    def link_predict(z, edge_label_index):
        pred = z[edge_label_index[0]] * z[edge_label_index[1]]
        assert len(pred) == edge_label_index.shape[1], (len(pred), edge_label_index.shape[1])
        return pred

    @classmethod
    def ff_loss(cls, out, edge_label, theta):
        pos_edge_mask, neg_edge_mask = edge_label == 1, edge_label == 0
        out_pos, out_neg = out[pos_edge_mask], out[neg_edge_mask]

        loss_pos, cumulated_logits_pos = forwardforward_loss_fn(out_pos, theta, target=1.0)
        loss_neg, cumulated_logits_neg = forwardforward_loss_fn(out_neg, theta, target=0.0)
        loss = loss_pos + loss_neg

        return loss, (out_pos, out_neg), (cumulated_logits_pos, cumulated_logits_neg)

    def forwardlearn_loss(self, out, edge_label):
        loss = self.criterion_no_reduction(out.sum(dim=1), edge_label)  # shape=(# pos and neg edges,)
        loss_mean = loss.mean()

        pos_edge_mask, neg_edge_mask = edge_label == 1, edge_label == 0
        with torch.no_grad():
            cumulated_logits_pos = loss[pos_edge_mask].exp().mean().item()
            cumulated_logits_neg = (1 - loss[neg_edge_mask].exp()).mean().item()

        return loss_mean, (out[pos_edge_mask], out[neg_edge_mask]), (cumulated_logits_pos, cumulated_logits_neg)

    def forward_train(
            self,
            train_data: Data,
            theta: float,
            **kwargs,
    ):
        self.optimizer.zero_grad()

        z = self.forward(train_data.x, train_data.edge_index)
        out = self.link_predict(z, train_data.edge_label_index)  # shape=(# pos and neg edges, node-emb-dim)

        if self.forward_learning_type == "FL":
            loss, out_tuple, logit_tuple = self.forwardlearn_loss(out, train_data.edge_label)
        elif self.forward_learning_type == "FF":
            loss, out_tuple, logit_tuple = self.ff_loss(out, train_data.edge_label, theta=theta)
        else:
            raise ValueError(f"Undefined: {self.forward_learning_type}")

        loss.backward()
        self.optimizer.step()

        return out_tuple, logit_tuple

    @torch.no_grad()
    def forward_predict(
            self,
            input_feats: torch.Tensor,
            edge_index: torch.Tensor,
            edge_label_index: torch.Tensor,
            theta: float,
    ):
        """Evaluate the layer with the given input and theta."""
        node_emb = self.forward(input_feats, edge_index)
        out = self.link_predict(node_emb, edge_label_index)

        if self.forward_learning_type == "FL":
            edge_score = out.sum(dim=1).sigmoid()
        elif self.forward_learning_type == "FF":
            logits = torch.nan_to_num(out.pow(2).sum(dim=1) - theta)
            edge_score = logits.sigmoid()
        else:
            raise ValueError(f"Undefined: {self.forward_learning_type}")

        return node_emb, edge_score


class GNNForward(torch.nn.Module):
    def __init__(
            self,
            layer_sizes: list,
            optimizer_name: str,
            optimizer_kwargs: dict,
            forward_learning_type: str,
            gnn: str,
            args: argparse.Namespace,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(LayerNormalization())

            self.layers.append(
                GNNForwardLayer(
                    gnn_layer=GNNConv(
                        gnn=gnn, in_channels=layer_sizes[i],
                        out_channels=layer_sizes[i + 1],
                    ),
                    optimizer_name=optimizer_name,
                    optimizer_kwargs=optimizer_kwargs,
                    forward_learning_type=forward_learning_type,
                    args=args,
                )
            )
        self.theta = args.ff_theta
        self.patience = args.patience
        self.val_every = args.val_every
        self.args = args

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

    def forward_train(
            self,
            train_data: torch_geometric.data.Data,
            val_data: torch_geometric.data.Data,
            test_data: torch_geometric.data.Data,
            result_manager: ResultManager,
            run_i: int,
    ):
        self.train()
        start = timer()

        train_data = train_data.clone().to(self.args.device)
        val_data = val_data.clone().to(self.args.device)
        test_data = test_data.clone().to(self.args.device)

        train_epochs, best_val_epochs = [], []
        get_perf_dict = partial(load_perf_dict, start, train_epochs, best_val_epochs)

        for i, layer in enumerate(self.layers):
            if layer.requires_training:
                early_stopper = EarlyStopping(self.patience) if self.patience >= 0 else None
                perf_manager = PerformanceManager("AUC")
                epoch_tqdm, epoch = tqdm(range(self.args.epochs)), -1
                try:
                    for epoch in epoch_tqdm:
                        self.train()

                        _, logits = layer.forward_train(
                            train_data,
                            self.theta,
                            epoch=epoch,
                        )

                        """Validation and Testing"""
                        if early_stopper is not None and (epoch + 1) % self.val_every == 0 \
                                and epoch > self.args.val_from:
                            val_auc, _ = self.eval_model(eval_data=val_data, last_eval_layer=i, print_eval=False)
                            if perf_manager.update_val_perf(val_auc, epoch):
                                test_auc, _ = self.eval_model(eval_data=test_data, last_eval_layer=i, print_eval=False)
                                perf_manager.update_test_perf(test_auc, epoch)

                            if early_stopper.step(val_auc, layer):
                                print(f"[Layer {i}: Epoch-{epoch}] Early stop!")
                                break

                        separation = logits[0] - logits[1]
                        epoch_tqdm.set_description(
                            f"[Layer {i}: Epoch-{epoch}] "
                            f"Goodness: pos={logits[0]:.4f}, neg={logits[1]:.4f} | "
                            f"Separation: {separation:.4f} | "
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

                test_auc, _ = self.eval_model(eval_data=test_data, last_eval_layer=i, print_eval=False)
                print(f"[Layer-{i}] Test AUC : {test_auc:.6f}\n")
                assert (i + 1) % 2 == 0, i
                result_manager.save_run_result(run_i, perf_dict=get_perf_dict(perf=test_auc), num_layers=(i + 1) // 2)  # save as layer 1, 2, 3, ...

            with torch.no_grad():
                if layer.requires_training:
                    # create input features for the next layer
                    train_data.x = layer(train_data.x, train_data.edge_index).detach()
                else:  # normalization layer
                    assert isinstance(layer, LayerNormalization), layer
                    train_data.x = layer(train_data.x).detach()

        logger.info("Finished training the network.")
        return get_perf_dict()

    @torch.no_grad()
    def eval_model(
            self,
            eval_data: Data,
            last_eval_layer: int = -1,
            print_eval=True,
    ):
        self.eval()

        if last_eval_layer < 0:
            last_eval_layer = len(self.layers) - 1

        accumulated_goodness_list = []
        node_feats = eval_data.x
        accumulate_from = 0

        for i, layer in enumerate(self.layers):
            if i <= last_eval_layer:
                node_feats, edge_score = layer.forward_predict(
                    node_feats, eval_data.edge_index, eval_data.edge_label_index, self.theta
                )

                if i >= accumulate_from:
                    accumulated_goodness_list.append(edge_score)

        accumulated_goodness = torch.stack(accumulated_goodness_list).mean(dim=0)
        eval_perf_dict = eval_link_prediction(
            y_true=eval_data.edge_label,
            y_score=accumulated_goodness,
        )
        rocauc = eval_perf_dict['rocauc']
        if print_eval:
            print("\nTest ROCAUC: {:.6f}".format(rocauc))

        return rocauc, accumulated_goodness
