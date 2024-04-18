# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score


def eval_node_classification(y_true, y_pred, mask):
    num_correct = y_pred[mask] == y_true[mask]
    # noinspection PyUnresolvedReferences
    acc = int(num_correct.sum()) / int(mask.sum())
    return {'acc': acc}


def eval_link_prediction(y_true, y_score, metrics=None):
    assert torch.is_tensor(y_true) and torch.is_tensor(y_score), (y_true, y_score)
    assert y_true.ndim == y_score.ndim == 1, (y_true.shape == y_score.shape)
    assert torch.all(y_score <= 1.0) and torch.all(y_score >= 0.0)

    y_true = y_true.detach().cpu().numpy()
    y_score = y_score.detach().cpu().numpy()

    if metrics is None:
        metrics = ['rocauc', 'ap', 'ndcg']
    if not isinstance(metrics, list):
        metrics = [metrics]

    perf_dict = {}
    for metric in metrics:
        if metric == 'rocauc':
            # roc auc: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
            perf_dict['rocauc'] = roc_auc_score(y_true, y_score)
        elif metric == 'ap':  # average precision
            # average precision: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
            perf_dict['ap'] = average_precision_score(y_true, y_score)
        elif metric == 'ndcg':
            # ndcg: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html
            perf_dict['ndcg'] = ndcg_score(np.asarray([y_true]), np.asarray([y_score]), ignore_ties=True)
        else:
            raise ValueError(f"Invalid metric: {metric}")

    return perf_dict
