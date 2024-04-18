# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy
import json
import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


class SeedManager:
    def __init__(self, seed_base):
        self.seed_base = seed_base
        self.run_i: Optional[int] = None

    def set_run_i(self, run_i):
        self.run_i = run_i

    def get_run_seed(self, run_i=None):
        if run_i is None:
            run_i = self.run_i
        return self.seed_base * 101 + run_i * 3


def setup_cuda(args):
    args.cuda = args.gpu >= 0 and torch.cuda.is_available()
    args.device = torch.device("cuda:{}".format(args.gpu) if args.cuda else "cpu")
    if args.cuda:
        torch.cuda.set_device(args.device)
    return args


class EarlyStopping:
    def __init__(self,
                 patience=30,
                 is_minimizing_objective=False,
                 logging=True,
                 score_type='score'):
        self.patience = patience
        self.is_minimizing_objective = is_minimizing_objective
        self.counter = 0
        self.early_stop = False
        self.logging = logging
        self.best_score = None
        self.best_model_state_dict = None
        self.score_type = score_type

    def step(self, score, model=None):
        """Return whether to early stop"""
        if self.best_score is None or self.improved(score, self.best_score):
            self.best_score = score
            if model is not None:
                self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def improved(self, score, best_score):
        if self.is_minimizing_objective:
            return True if score < best_score else False
        else:
            return True if score > best_score else False

    def save_checkpoint(self, model):
        self.best_model_state_dict = copy.deepcopy(model.state_dict())

    def load_checkpoint(self, model):
        model.load_state_dict(self.best_model_state_dict)


class ResultManager:
    def __init__(self,
                 result_file_prefix,
                 args,
                 seed_manager,
                 decimals=6,
                 delimiter=","):
        self.result_file_prefix = result_file_prefix
        self.args = args
        self.num_layers = self.args.num_layers
        self.seed_manager = seed_manager
        self.decimals = decimals
        self.delimiter = delimiter

    def run_result_path(self, run_i, num_layers=None):
        if num_layers is None:
            num_layers = self.num_layers
        seed_i = self.seed_manager.get_run_seed(run_i)
        run_result_path = self.args.results_dir / \
                          f"{self.result_file_prefix}-{self.args.task}-{self.args.model}-" \
                          f"num_layers{num_layers}-run{run_i}-seed{seed_i}.json"
        return run_result_path

    def save_run_result(self, run_i, perf_dict, num_layers=None):
        result_dict = {
            'perf': np.round(perf_dict['perf'], self.decimals),
            'best_val_epoch': "-".join(map(str, perf_dict['best_val_epochs'])),
            'train_epochs': "-".join(map(str, perf_dict['train_epochs'])),
            'train_time': np.round(perf_dict['train_time'], self.decimals),
            'run_i': run_i,
            'run_seed': self.seed_manager.get_run_seed(run_i),
        }
        args_dict = vars(copy.deepcopy(self.args))
        del args_dict['results_dir']
        args_dict['device'] = str(args_dict['device'])
        if num_layers is not None:
            args_dict['num_layers'] = num_layers
        result_dict.update(args_dict)

        with self.run_result_path(run_i, num_layers).open('w') as f:
            json.dump(result_dict, f, indent=2)

    def load_run_result(self, run_i, num_layers=None):
        try:
            with self.run_result_path(run_i, num_layers).open('r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
