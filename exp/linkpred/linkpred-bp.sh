#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

scriptDir=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "${scriptDir}"/../../src/ || exit

EXP_SETTING='perf-results'
TASK='link-pred'

NUM_RUNS=5
SEED=100
EPOCHS=1000
VAL_EVERY=2
PATIENCE=100
NUM_HIDDEN=128
LR=0.001

for dataset in "CitationFull-CiteSeer" "CitationFull-Cora_ML" "CitationFull-PubMed" "Amazon-Photo" "GitHub"; do
  DATASET="${dataset}"

for model in "GNN-GCN" "GNN-SAGE" "GNN-GAT"; do

for num_layers in 1 2 3 4; do

python train_backprop.py \
--exp-setting "${EXP_SETTING}" \
--dataset "${DATASET}" \
--task "${TASK}" \
--model "${model}" \
--num-layers "${num_layers}" \
--num-runs "${NUM_RUNS}" --seed "${SEED}" --epochs "${EPOCHS}" --val-every "${VAL_EVERY}" \
--lr "${LR}" --patience "${PATIENCE}" --num-hidden "${NUM_HIDDEN}" \
--overwrite-result "$@"

done

done

done