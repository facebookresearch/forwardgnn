# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import traceback
from functools import partial
from timeit import default_timer as timer
from typing import List

import torch
import torch_geometric
from tqdm import tqdm

from forward_learning.loss import forwardforward_loss_fn
from forward_learning.nodeclass.common import LayerNormalization
from forward_learning.utils import PerformanceManager, load_perf_dict
from gnn import GNNConv, CachedSAGEConv, CachedGCNConv
from utils import logger, ResultManager, EarlyStopping


class GNNFFLayer(torch.nn.Module):
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

    @property
    def requires_training(self):
        return True

    def forward(self, x, edge_index):
        return self.gnn_layer(x, edge_index)

    def clear_cached_propagate(self):
        if isinstance(self.gnn_layer.gnn, (CachedSAGEConv, CachedGCNConv)):
            self.gnn_layer.gnn.clear_cached()

    def forward_train(
        self,
        train_mask,
        pos_feats: torch.Tensor,
        neg_feats_list: List[torch.Tensor],
        theta: float,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
    ):
        pos_target = torch.ones(train_mask.sum(), device=pos_feats.device)
        neg_target = torch.zeros(train_mask.sum(), device=pos_feats.device)

        y_pos = self.forward(pos_feats, pos_edge_index)

        y_neg_list = []
        for i, neg_feats in enumerate(neg_feats_list):
            y_neg = self.forward(neg_feats, neg_edge_index)
            y_neg_list.append(y_neg)

        if self.loss_fn_name == "forwardforward_loss_fn":
            loss_pos, cumulated_logits_pos = forwardforward_loss_fn(y_pos[train_mask], theta, pos_target)
        else:
            raise ValueError(f"Undefined: {self.loss_fn_name}")

        cumulated_logits_neg_sum, num_total_negs = 0, 0
        loss_neg_parts = []
        for i, y_neg in enumerate(y_neg_list):
            if self.loss_fn_name == "forwardforward_loss_fn":
                loss_neg_part, cumulated_logits_neg_part = forwardforward_loss_fn(y_neg[train_mask], theta, neg_target)
            else:
                raise ValueError(f"Undefined: {self.loss_fn_name}")

            num_train_nodes = train_mask.sum().item()
            cumulated_logits_neg_sum += cumulated_logits_neg_part * num_train_nodes
            num_total_negs += num_train_nodes
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
        self, input_feats: torch.Tensor, theta: float, edge_index: torch.Tensor
    ):
        """Evaluate the layer with the given input and theta."""
        y = self.forward(input_feats, edge_index)
        return y, torch.square(y).mean(dim=1) - theta


class GNNLabelAppendingFF(torch.nn.Module):
    def __init__(
        self,
        layer_sizes: list,
        optimizer_name: str,
        optimizer_kwargs: dict,
        loss_fn_name: str,
        num_classes: int,
        gnn: str,
        args: argparse.Namespace,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(LayerNormalization())

            in_channels = layer_sizes[i] + num_classes if i == 0 else layer_sizes[i]
            self.layers.append(
                GNNFFLayer(
                    gnn_layer=GNNConv(
                        gnn=gnn, in_channels=in_channels,
                        out_channels=layer_sizes[i + 1]
                    ),
                    optimizer_name=optimizer_name,
                    optimizer_kwargs=optimizer_kwargs,
                    loss_fn_name=loss_fn_name,
                    args=args,
                )
            )
        self.num_classes = num_classes
        self.label_appender = GNNLabelAppender(
            device=args.device,
            label_names=list(range(self.num_classes)),
            zero_or_uniform_for_unlabeled_nodes="uniform",
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
            data: torch_geometric.data.Data,
            result_manager: ResultManager,
            run_i: int
    ):
        self.train()
        start = timer()

        data = data.to(self.args.device)
        pos_feats, neg_feats_list = self.label_appender.append_for_train(
            data.x, data.y, train_mask=data.train_mask, num_negs=self.args.num_negs
        )
        pos_edge_index, neg_edge_index = data.edge_index, data.edge_index

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
                            data.train_mask,
                            pos_feats, neg_feats_list,
                            self.theta,
                            pos_edge_index, neg_edge_index,
                        )

                        """Validation and Testing"""
                        if early_stopper is not None and (epoch + 1) % self.val_every == 0 and epoch > self.args.val_from:
                            val_acc, _ = self.eval_model(data, train_mask=data.train_mask, eval_mask=data.val_mask,
                                                         last_eval_layer=i, print_eval=False)
                            if perf_manager.update_val_perf(val_acc, epoch):
                                test_acc, _ = self.eval_model(data, train_mask=data.train_mask | data.val_mask,
                                                              eval_mask=data.test_mask, last_eval_layer=i,
                                                              print_eval=False)
                                perf_manager.update_test_perf(test_acc, epoch)

                            if early_stopper.step(val_acc, layer):
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

                test_acc, _ = self.eval_model(data, train_mask=data.train_mask | data.val_mask,
                                              eval_mask=data.test_mask, last_eval_layer=i, print_eval=False)
                print(f"[Layer-{i}] Test Accuracy : {test_acc:.4f}%\n")
                assert (i + 1) % 2 == 0, i
                result_manager.save_run_result(run_i, perf_dict=get_perf_dict(perf=test_acc), num_layers=(i + 1) // 2)  # save as layer 1, 2, 3, ...

            with torch.no_grad():
                if layer.requires_training:
                    # create input features for the next layer
                    pos_feats = layer(pos_feats, pos_edge_index).detach()
                    neg_feats_list = [layer(neg_feats, neg_edge_index).detach() for neg_feats in neg_feats_list]
                else:  # normalization layer
                    assert isinstance(layer, LayerNormalization), layer
                    pos_feats = layer(pos_feats)
                    neg_feats_list = [layer(neg_feats) for neg_feats in neg_feats_list]

        logger.info("Finished training the network.")
        return get_perf_dict()

    @torch.no_grad()
    def eval_model(
            self,
            data: torch_geometric.data.Data,
            *,
            train_mask: torch.BoolTensor,
            eval_mask: torch.BoolTensor,
            last_eval_layer: int = -1,
            print_eval=True
    ):
        """Evaluate the network with the given input and theta."""
        self.eval()

        data = data.to(self.args.device)

        accumulated_goodness_list = []
        if last_eval_layer < 0:
            last_eval_layer = len(self.layers) - 1
        accumulate_from = 0

        for eval_class_label in range(self.num_classes):
            input_feats = self.label_appender.append_for_eval(
                eval_class_label=eval_class_label, node_feats=data.x, node_labels=data.y,
                train_mask=train_mask, eval_mask=eval_mask
            )
            accumulated_goodness = torch.zeros(input_feats.shape[0], device=input_feats.device)

            for i, layer in enumerate(self.layers):
                if i <= last_eval_layer:
                    input_feats, goodness = layer.forward_predict(
                        input_feats, self.theta, data.edge_index
                    )
                    if i >= accumulate_from:
                        accumulated_goodness += goodness

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


class GNNLabelAppender:
    def __init__(
            self,
            label_names: List,
            device,
            zero_or_uniform_for_unlabeled_nodes: str = 'uniform'
    ):
        self.label_names = label_names
        self.one_hot_labels = torch.stack([
            torch.nn.functional.one_hot(torch.tensor(i), len(label_names))
            for i in range(len(label_names))
        ]).float().to(device)
        self.device = device
        self.zero_or_uniform_for_unlabeled_nodes = zero_or_uniform_for_unlabeled_nodes
        assert self.zero_or_uniform_for_unlabeled_nodes in ['zero', 'uniform'], zero_or_uniform_for_unlabeled_nodes

    @torch.no_grad()
    def append_for_train(
            self,
            node_feats: torch.Tensor,
            node_labels: torch.Tensor,
            train_mask: torch.BoolTensor,
            num_negs=1
    ):
        """
        append one hot labels to features of training nodes, and uniform (or zero) distribution vectors as features to all other nodes
        """
        num_nodes = node_labels.shape[0]
        num_classes = self.one_hot_labels.shape[1]
        num_negs = min(num_negs, num_classes - 1)

        """Positive feats"""
        positive_appending_labels = torch.zeros((num_nodes, num_classes)).to(self.device)
        positive_appending_labels[train_mask] = self.one_hot_labels[node_labels[train_mask]]
        if self.zero_or_uniform_for_unlabeled_nodes == 'uniform':  # uniform distribution for non-training nodes
            positive_appending_labels[~train_mask] = \
                torch.full((num_classes,), fill_value=1 / num_classes).view(1, num_classes).repeat((~train_mask).sum(), 1).to(self.device)
        else:  # all zeros for non-training nodes
            positive_appending_labels[~train_mask] = \
                torch.full((num_classes,), fill_value=0.0).view(1, num_classes).repeat((~train_mask).sum(), 1).to(self.device)
        positive_feats = torch.cat([node_feats, positive_appending_labels], dim=1)

        """Negative feats"""
        # randomly select negative classes for each node
        x = torch.arange(num_classes).repeat(num_nodes, 1)  # shape=(num nodes, num classes)
        # ref: https://discuss.pytorch.org/t/what-is-the-most-efficient-way-to-shuffle-each-row-of-a-tensor-with-different-shuffling-order-for-each-of-the-row/109772
        r = torch.rand_like(x.float())
        r[torch.arange(num_nodes), node_labels.cpu()] = 10000000  # to exclude the true class of each node
        assert num_classes < 10000000
        indices = torch.argsort(r, dim=-1, descending=False)
        neg_classes = torch.gather(x, dim=-1, index=indices)[:, 0:-1]  # shape=(num nodes, num classes - 1). 0:-1 to exclude the true class
        neg_classes = neg_classes.to(self.device)

        negative_feats_list = []
        for i in range(min(num_negs, neg_classes.shape[1])):
            negative_appending_labels = torch.zeros((num_nodes, num_classes)).to(self.device)

            neg_classes_i = neg_classes[:, i:i + 1].view(-1)
            negative_appending_labels[train_mask] = self.one_hot_labels[neg_classes_i[train_mask]]

            if self.zero_or_uniform_for_unlabeled_nodes == 'uniform':  # uniform distribution for non-training nodes
                negative_appending_labels[~train_mask] = \
                    torch.full((num_classes,), fill_value=1 / num_classes).view(1, num_classes).repeat((~train_mask).sum(), 1).to(self.device)
            else:  # all zeros for non-training nodes
                negative_appending_labels[~train_mask] = \
                    torch.full((num_classes,), fill_value=0.0).view(1, num_classes).repeat((~train_mask).sum(), 1).to(self.device)

            negative_feats = torch.cat([node_feats, negative_appending_labels], dim=1)
            negative_feats_list.append(negative_feats)

        return positive_feats, negative_feats_list

    @torch.no_grad()
    def append_for_eval(
            self,
            eval_class_label: int,
            node_feats: torch.Tensor,
            node_labels: torch.Tensor,
            train_mask: torch.BoolTensor,
            eval_mask: torch.BoolTensor
    ):
        num_nodes = node_labels.shape[0]
        num_classes = self.one_hot_labels.shape[1]

        appending_labels = torch.zeros((num_nodes, num_classes)).to(self.device)
        appending_labels[train_mask] = self.one_hot_labels[node_labels[train_mask]]
        appending_labels[eval_mask] = self.one_hot_labels[eval_class_label].repeat(eval_mask.sum(), 1)

        if self.zero_or_uniform_for_unlabeled_nodes == 'uniform':
            appending_labels[~(train_mask | eval_mask)] = \
                torch.full((num_classes,), fill_value=1 / num_classes).repeat((~(train_mask | eval_mask)).sum(), 1).to(self.device)
        else:
            appending_labels[~(train_mask | eval_mask)] = \
                torch.full((num_classes,), fill_value=0.0).repeat((~(train_mask | eval_mask)).sum(), 1).to(self.device)

        return torch.cat([node_feats, appending_labels], dim=1)
