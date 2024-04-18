# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from timeit import default_timer as timer

import numpy as np


class PerformanceManager:
    def __init__(self, perf_metric):
        self.perf_metric = perf_metric

        self.best_val_perf = -np.inf
        self.best_val_epoch = -1
        self.latest_val_perf = None
        self.latest_epoch = -1

        self.best_test_perf = -np.inf
        self.best_test_epoch = -1
        self.latest_test_perf = None
        self.latest_epoch = -1

    def update_val_perf(self, val_perf, epoch):
        self.latest_val_perf = val_perf
        self.latest_epoch = epoch

        perf_improved = False
        if val_perf > self.best_val_perf:
            self.best_val_perf = val_perf
            self.best_val_epoch = epoch
            perf_improved = True
        return perf_improved

    def update_test_perf(self, test_perf, epoch):
        self.latest_test_perf = test_perf
        self.latest_epoch = epoch

        perf_improved = False
        if test_perf > self.best_test_perf:
            self.best_test_perf = test_perf
            self.best_test_epoch = epoch
            perf_improved = True
        return perf_improved

    def val_perf_summary(self):
        if self.best_val_epoch == -1:
            return ""

        perf_summary = f"Val: {self.perf_metric}={self.latest_val_perf:.2f}"
        if self.best_val_epoch >= 0:
            perf_summary += f" (Best{self.perf_metric}={self.best_val_perf:.4f}@Epoch-{self.best_val_epoch})"
        return perf_summary

    def test_perf_summary(self):
        if self.best_test_epoch == -1:
            return ""

        perf_summary = f"Test: {self.perf_metric}={self.latest_test_perf:.2f}"
        if self.best_test_epoch >= 0:
            perf_summary += f" (Best{self.perf_metric}={self.best_test_perf:.4f}@Epoch-{self.best_test_epoch})"
        return perf_summary


def load_perf_dict(start_time, train_epochs, best_val_epochs, perf=None):
    perf_dict = {
        "train_time": timer() - start_time,
        "train_epochs": list(train_epochs),
        "best_val_epochs": list(best_val_epochs),
    }
    if perf is not None:
        perf_dict["perf"] = perf
    return perf_dict
