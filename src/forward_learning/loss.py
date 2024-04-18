# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F


def forwardforward_loss_fn(y, theta, target):  # target=1 if positive, 0 otherwise.
    if isinstance(target, (int, float)):  # if sign (1 or -1) is given, turn it into a target (1 or 0)
        target = max(0.0, float(target))

    if isinstance(target, (int, float)):
        target = torch.tensor([target] * len(y), device=y.device, dtype=torch.float)

    logits = y.pow(2).sum(dim=1) - theta
    with torch.no_grad():
        accumulated_logits = logits.mean().item()

    loss = F.binary_cross_entropy_with_logits(input=logits, target=target, reduction='mean')
    return loss, accumulated_logits
