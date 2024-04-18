#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

scriptDir=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "${scriptDir}"/../../src/ || exit

EXP_SETTING='perf-results'
TASK='node-class'

NUM_RUNS=5
SEED=100
EPOCHS=1000
NUM_HIDDEN=128
LR=0.001
NUM_NEGS=100  # to use all negatives
LOSS_FN_NAME=forwardforward_loss_fn
VAL_EVERY=2
PATIENCE=100
declare -a DATASETS=("CitationFull-CiteSeer" "CitationFull-Cora_ML" "CitationFull-PubMed" "Amazon-Photo" "GitHub")

for model in "GNN_LA-GCN" "GNN_LA-SAGE" "GNN_LA-GAT"; do

for dataset in "${DATASETS[@]}"; do
  DATASET="${dataset}"

for num_layers in 4; do

python train_forward.py \
--exp-setting "${EXP_SETTING}" \
--dataset "${DATASET}" \
--task "${TASK}" \
--model "${model}" \
--num-layers "${num_layers}" \
--num-runs "${NUM_RUNS}" --seed "${SEED}" --epochs "${EPOCHS}" --val-every "${VAL_EVERY}" \
--lr "${LR}" --patience "${PATIENCE}" --num-hidden "${NUM_HIDDEN}" \
--loss-fn-name "${LOSS_FN_NAME}" \
--num-negs "${NUM_NEGS}" \
--overwrite-result "$@"

done

done

done