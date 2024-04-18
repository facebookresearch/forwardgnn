#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

scriptDir=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "${scriptDir}"/ || exit

CONDA_ENV=ForwardLearningGNN

if conda info --envs | grep -q "${CONDA_ENV} "; then
  echo "\"${CONDA_ENV}\" conda env exists.";
else
  conda create -y --name "${CONDA_ENV}" python=3.8
fi

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}"/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

if [[ "${OSTYPE}" == "darwin"* ]]; then  # Mac OS
  conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
else
  conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
fi

conda install -y pyg==2.2.0 -c pyg  # for Linux and OSX
conda install -y -c conda-forge tqdm