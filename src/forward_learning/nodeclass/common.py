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
from typing import Optional

import torch
import torch.utils.data


# noinspection PyUnusedLocal
class LayerNormalization(torch.nn.Module):
    """Applies row wise normalization"""
    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(
            self,
            x,
            edge_index: Optional[torch.Tensor] = None
    ):
        l2_norm = (
            torch.norm(x.reshape(x.shape[0], -1), p=2, dim=1, keepdim=True)
            + 1e-8
        )
        return x / l2_norm

    def forward_train(
            self,
            input_tensor: torch.Tensor,
            signs: torch.Tensor,
            theta: float,
            pos_edge_index: Optional[torch.Tensor] = None,
            neg_edge_index: Optional[torch.Tensor] = None
    ):
        with torch.no_grad():
            output = self()
        return output, None

    @torch.no_grad()
    def forward_predict(
            self,
            input_feats: torch.Tensor,
            theta: float,
            edge_index: Optional[torch.Tensor] = None,
            edge_type: Optional[torch.Tensor] = None
    ):
        with torch.no_grad():
            output = self(input_feats)

        return output, torch.zeros(input_feats.shape[0], device=input_feats.device)

    @property
    def requires_training(self):
        return False
