# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(module)s.%(funcName)s] %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class Logger(object):
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        self.log.flush()


def log_stdout(log_file):
    if sys.platform == "darwin":  # OS X
        sys.stdout = Logger(log_file)
    else:
        sys.stdout = sys.stderr = Logger(log_file)

    fileHandler = logging.FileHandler(log_file)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    print(f"stdout being logged to {log_file}")
