# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from collections import OrderedDict
from datetime import datetime
from timeit import default_timer as timer

import torch
import numpy as np
from torch_geometric.nn import GraphSAGE, GCN, GAT
from tqdm import tqdm

import settings
from datasets import load_node_classification_data, load_link_prediction_data
from utils.train_utils import setup_cuda, set_seed, EarlyStopping, SeedManager, ResultManager
from utils.log_utils import log_stdout, logger
from utils.eval_utils import eval_node_classification, eval_link_prediction


def main(args):
    seed_manager = SeedManager(args.seed)
    result_manager = ResultManager(result_file_prefix=f"bp-results", args=args, seed_manager=seed_manager)

    for run_i in range(args.num_runs):
        if result_manager.load_run_result(run_i) is not None and not args.overwrite_result:
            logger.info(f"Skipping run-{run_i}: already evaluated.")
            continue

        seed_manager.set_run_i(run_i)
        set_seed(seed=seed_manager.get_run_seed(), deterministic="gat" not in args.model.lower())  # initialize seed for each run
        print(f"\nStarting run-{run_i} of {args.model} on {args.dataset} (seed={seed_manager.get_run_seed()})\n")

        if args.task == "node-class":
            data = load_node_classification_data(args, split_i=run_i)
            model = build_model(data, args)
            trainer = NodeClassificationTrainer(
                model, data, args.device, args.lr, args.epochs, args.patience, args
            )
        elif args.task == "link-pred":
            train_data, val_data, test_data, data = load_link_prediction_data(args, split_i=run_i)
            model = build_model(data, args)
            trainer = LinkPredictionTrainer(
                model, train_data, val_data, test_data, args.device, args.lr, args.epochs, args.patience, args
            )
        else:
            raise ValueError(f"Invalid task: {args.task}")

        """Training and testing"""
        result = trainer.train_test()

        perf_dict = {
            "perf": result["test_perf"],
            "train_time": result["train_time"],
            "train_epochs": [result["train_epochs"]],
            "best_val_epochs": [-1],
        }
        result_manager.save_run_result(run_i, perf_dict)

    perfs = [result_manager.load_run_result(run_i)['perf'] for run_i in range(args.num_runs)]
    print(f"\nTest Performance ({args.num_runs} runs): {np.mean(perfs):.6f}%±{np.std(perfs):.4f}")


def build_model(data, args):
    assert args.num_layers >= 1, args.num_layers
    if args.task == "node-class":
        out_channels = data.num_classes
    else:
        out_channels = None  # will be set to args.num_hidden

    if args.model == "GNN-SAGE":
        model = GraphSAGE(
            in_channels=data.num_features,
            hidden_channels=args.num_hidden,
            num_layers=args.num_layers,
            out_channels=out_channels,
            dropout=0.0,
            act="relu"
        )
    elif args.model == "GNN-GCN":
        model = GCN(
            in_channels=data.num_features,
            hidden_channels=args.num_hidden,
            num_layers=args.num_layers,
            out_channels=out_channels,
            dropout=0.0,
            act="relu"
        )
    elif args.model == "GNN-GAT":
        model = GAT(
            in_channels=data.num_features,
            hidden_channels=args.num_hidden,
            heads=4,
            num_layers=args.num_layers,
            out_channels=out_channels,
            dropout=0.0,
            act="relu"
        )
    else:
        raise ValueError(f"Invalid model: {args.model}")

    print(model)
    return model


class NodeClassificationTrainer:
    def __init__(self, model, data, device, lr, epochs, patience, args):
        self.model = model
        self.data = data
        self.device = device

        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.args = args

    def train(self):
        model, data = self.model, self.data

        """Training"""
        start = timer()
        stopper = EarlyStopping(self.patience) if self.patience >= 0 else None
        epoch = -1

        data = data.clone().to(self.device)
        model = model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        epoch_tqdm = tqdm(range(self.epochs))
        for epoch in epoch_tqdm:
            optimizer.zero_grad()

            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])

            loss.backward()
            optimizer.step()

            """Validation"""
            if stopper is not None and (epoch + 1) % self.args.val_every == 0:
                val_metric = 'acc'

                with torch.no_grad():
                    val_out = model(data.x, data.edge_index)

                pred = val_out.argmax(dim=1)
                val_perf_dict = eval_node_classification(data.y, pred, data.val_mask)
                val_perf = val_perf_dict[val_metric]

                epoch_tqdm.set_description(
                    f'Epoch: {epoch:03d}, Train Loss={loss.item():.4f}, Val Acc={val_perf:.4f}'
                )

                if stopper.step(val_perf, model):
                    print(f"[Epoch-{epoch}] Early stop!")
                    break

        if stopper is not None and stopper.best_score is not None:
            stopper.load_checkpoint(model)

        train_time = timer() - start
        logger.info("Finished training")

        return train_time, epoch

    def test(self):
        model, data = self.model, self.data.to(self.device)
        model.eval()

        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_perf_dict = eval_node_classification(data.y, pred, data.test_mask)
        test_acc = test_perf_dict['acc']
        print(f"Test Accuracy: {test_acc:.6f}")

        return test_acc

    def train_test(self):
        train_time, train_epochs = self.train()
        test_acc = self.test()

        return {
            "test_perf": test_acc,
            "train_time": train_time,
            "train_epochs": train_epochs,
        }


class LinkPredictionTrainer:
    def __init__(self, model, train_data, val_data, test_data, device, lr, epochs, patience, args):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.device = device

        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.args = args

    @staticmethod
    def link_predict(z, edge_label_index):
        pred = (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=1)
        assert len(pred) == edge_label_index.shape[1], (len(pred), edge_label_index.shape[1])
        return pred

    def train(self):
        model, train_data, val_data, test_data = self.model, self.train_data, self.val_data, self.test_data
        model.train()

        """Training"""
        start = timer()
        stopper = EarlyStopping(self.patience) if self.patience >= 0 else None
        epoch = -1

        train_data = train_data.clone().to(self.device)
        val_data = val_data.clone().to(self.device)
        model = model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=5e-4)
        criterion = torch.nn.BCEWithLogitsLoss()

        epoch_tqdm = tqdm(range(self.epochs))
        for epoch in epoch_tqdm:
            model.train()

            optimizer.zero_grad()

            z = model(train_data.x, train_data.edge_index)
            out = self.link_predict(z, train_data.edge_label_index)
            loss = criterion(out, train_data.edge_label)

            loss.backward()
            optimizer.step()

            """Validation"""
            if stopper is not None and (epoch + 1) % self.args.val_every == 0:
                val_metric = 'rocauc'

                with torch.no_grad():
                    model.eval()
                    val_z = model(val_data.x, val_data.edge_index)
                    val_out = self.link_predict(val_z, val_data.edge_label_index)

                val_perf_dict = eval_link_prediction(
                    y_true=val_data.edge_label,
                    y_score=val_out.sigmoid(),
                    metrics=val_metric
                )
                val_perf = val_perf_dict[val_metric]
                epoch_tqdm.set_description(f'Epoch: {epoch:03d}, Train Loss={loss.item():.4f}, Val AUC={val_perf:.4f}')

                if stopper.step(val_perf, model):
                    print(f"[Epoch-{epoch}] Early stop!")
                    break

        if stopper is not None and stopper.best_score is not None:
            stopper.load_checkpoint(model)

        train_time = timer() - start
        logger.info("Finished training")

        return train_time, epoch

    def test(self):
        model, test_data = self.model, self.test_data.to(self.device)
        model.eval()

        z = model(test_data.x, test_data.edge_index)
        out = self.link_predict(z, test_data.edge_label_index)

        test_perf_dict = eval_link_prediction(
            y_true=test_data.edge_label,
            y_score=out.sigmoid(),
        )
        test_rocauc = test_perf_dict['rocauc']
        print(f"Test AUC: {test_rocauc:.6f}")

        return test_rocauc

    def train_test(self):
        train_time, train_epochs = self.train()
        test_rocauc = self.test()

        return {
            "test_perf": test_rocauc,
            "train_time": train_time,
            "train_epochs": train_epochs,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="node-class", choices=["link-pred", "node-class"],
                        help="graph learning task")
    parser.add_argument('--model', type=str, choices=["GNN-SAGE", "GNN-GCN", "GNN-GAT"],
                        help="model type")
    parser.add_argument('--dataset', type=str,
                        help="dataset name")
    parser.add_argument('--num-runs', type=int, default=1,
                        help="number of total runs. each run uses a different random seed.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--seed", type=int, default=100,
                        help="seed for exp")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--val-every", type=int, default=10,
                        help="number of epochs between validation")
    parser.add_argument('--patience', type=int, default=30,
                        help='patience for early stopping (set this to negative value to not use early stopping)')
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of gnn layers")
    parser.add_argument("--num-hidden", type=int, default=32,
                        help="number of hidden channels")
    parser.add_argument('--exp-setting', type=str, default="default",
                        help="experiment setting")
    parser.add_argument('--overwrite-result', action='store_true')
    parser.set_defaults(overwrite_result=False)

    args = parser.parse_args()
    return args


def populate_args(args):
    setup_cuda(args)

    args.results_dir = settings.RESULTS_ROOT / args.exp_setting / args.dataset / args.task
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.exp_datetime = datetime.today().strftime('%Y%m%d_%H%M%S')
    stdout_path = args.results_dir / "stdout"
    stdout_path.mkdir(parents=True, exist_ok=True)
    log_stdout(stdout_path / f"stdout-{args.task}-{args.model}-{args.dataset}-{args.exp_datetime}.txt")

    from pprint import pformat
    print(f"args:\n{pformat(args.__dict__)}")
    return args


if __name__ == '__main__':
    args = parse_args()
    args = populate_args(args)
    main(args)
