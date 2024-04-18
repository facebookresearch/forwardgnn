# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from pathlib import Path
ROOT_DIR_PATH = Path(os.path.dirname(__file__)) / '..'

DATA_ROOT = ROOT_DIR_PATH / 'data'
DATA_ROOT.mkdir(exist_ok=True)

RESULTS_ROOT = ROOT_DIR_PATH / 'results'
RESULTS_ROOT.mkdir(exist_ok=True)

DATASPLIT_ROOT = ROOT_DIR_PATH / 'datasplits'
DATASPLIT_ROOT.mkdir(exist_ok=True)
