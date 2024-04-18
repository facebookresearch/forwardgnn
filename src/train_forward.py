# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from datetime import datetime
from typing import Optional

import numpy as np

import settings
from datasets import load_node_classification_data, load_link_prediction_data
from forward_learning.linkpred import (
    GNNForward,
    GNNTowDownForward,
)
from forward_learning.nodeclass import (
    GNNLabelAppendingFF,
    GNNVirtualNodeLabelAppendingFF,
    GNNSingleForward,
    GNNTop2InputSF,
    GNNTop2LossSF,
)
from utils.log_utils import log_stdout, logger
from utils.train_utils import setup_cuda, set_seed, SeedManager, ResultManager


def main(args):
    seed_manager = SeedManager(args.seed)
    result_manager = ResultManager(result_file_prefix=f"fw-results", args=args, seed_manager=seed_manager)

    for run_i in range(args.num_runs):
        if not args.overwrite_result and result_manager.load_run_result(run_i) is not None:
            logger.info(f"Skipping run-{run_i}: already evaluated.")
            continue

        seed_manager.set_run_i(run_i)
        set_seed(seed_manager.get_run_seed(), deterministic="gat" not in args.model.lower())  # initialize seed for each run
        print(f"\nStarting run-{run_i} of {args.model} on {args.dataset} (seed={seed_manager.get_run_seed()})\n")

        # Node Classification
        if args.task == "node-class":
            data = load_node_classification_data(args, split_i=run_i)
            model = build_node_classification_model(model_type=args.model,
                                                    n_layers=args.num_layers,
                                                    hidden_size=args.num_hidden,
                                                    loss_fn_name=args.loss_fn_name,
                                                    lr=args.lr,
                                                    data=data,
                                                    args=args)
            model = model.to(args.device)

            """Training"""
            result = model.forward_train(data, result_manager, run_i)

            """Testing"""
            data = data.to(args.device)
            if isinstance(model, (GNNSingleForward,
                                  GNNTop2InputSF,
                                  GNNTop2LossSF)):
                acc, _ = model.eval_model(eval_mask=data.test_mask, print_eval=True)
            elif isinstance(model, (GNNLabelAppendingFF, GNNVirtualNodeLabelAppendingFF)):
                acc, _ = model.eval_model(data, train_mask=data.train_mask | data.val_mask, eval_mask=data.test_mask, print_eval=True)
            else:
                raise ValueError(f"Invalid model: {model}")
            perf = acc
        # Link Prediction
        elif args.task == "link-pred":
            train_data, val_data, test_data, data = load_link_prediction_data(args, split_i=run_i)
            model = build_link_prediction_model(model_type=args.model,
                                                n_layers=args.num_layers,
                                                hidden_size=args.num_hidden,
                                                loss_fn_name=args.loss_fn_name,
                                                lr=args.lr,
                                                data=data,
                                                args=args)
            model = model.to(args.device)

            """Training"""
            result = model.forward_train(train_data, val_data, test_data, result_manager, run_i)

            """Testing"""
            test_data = test_data.to(args.device)
            auc, _ = model.eval_model(test_data)
            perf = auc
        else:
            raise ValueError(f"Invalid task: {args.task}")

        perf_dict = {
            'perf': perf,
            'train_time': result["train_time"],
            'train_epochs': result["train_epochs"],
            'best_val_epochs': result["best_val_epochs"],
        }
        result_manager.save_run_result(run_i, perf_dict)

    perfs = [result_manager.load_run_result(run_i)['perf'] for run_i in range(args.num_runs)]
    print(f"\nTest Performance ({args.num_runs} runs): {np.mean(perfs):.6f}%±{np.std(perfs):.4f}")


def build_node_classification_model(model_type, n_layers, hidden_size, loss_fn_name, lr, data, args):
    assert n_layers >= 1, n_layers

    if "SingleForward".lower() in model_type.lower():
        if "Top".lower() in model_type.lower():
            if "Top2Input".lower() in model_type.lower():
                layer_sizes = [data.num_features] + [hidden_size] * n_layers + [data.num_classes]

                model = GNNTop2InputSF(
                    layer_sizes=layer_sizes,
                    optimizer_name='Adam',
                    optimizer_kwargs={'lr': lr, 'weight_decay': 5e-4},
                    num_classes=data.num_classes,
                    gnn=model_type.split("-")[-1],
                    args=args,
                )
            elif "Top2Loss".lower() in model_type.lower():
                layer_sizes = [data.num_features] + [hidden_size] * n_layers

                model = GNNTop2LossSF(
                    layer_sizes=layer_sizes,
                    optimizer_name='Adam',
                    optimizer_kwargs={'lr': lr, 'weight_decay': 5e-4},
                    num_classes=data.num_classes,
                    gnn=model_type.split("-")[-1],
                    args=args,
                )
            else:
                raise ValueError(f"Invalid model: {model_type}")
        else:
            layer_sizes = [data.num_features] + [hidden_size] * n_layers

            model = GNNSingleForward(
                layer_sizes=layer_sizes,
                optimizer_name='Adam',
                optimizer_kwargs={'lr': lr, 'weight_decay': 5e-4},
                num_classes=data.num_classes,
                gnn=model_type.split("-")[-1],
                args=args,
            )
    else:
        layer_sizes = [data.num_features] + [hidden_size] * n_layers

        if model_type.startswith("GNN_LA"):  # LA: Label Appending
            model = GNNLabelAppendingFF(
                layer_sizes=layer_sizes,
                optimizer_name='Adam',
                optimizer_kwargs={'lr': lr, 'weight_decay': 5e-4},
                loss_fn_name=loss_fn_name,
                num_classes=data.num_classes,
                gnn=model_type.split("-")[-1],
                args=args,
            )
        elif model_type.startswith("GNN_VNLA"):  # VN: Virtual Node
            model = GNNVirtualNodeLabelAppendingFF(
                layer_sizes=layer_sizes,
                optimizer_name='Adam',
                optimizer_kwargs={'lr': lr, 'weight_decay': 5e-4},
                loss_fn_name=loss_fn_name,
                num_classes=data.num_classes,
                gnn=model_type.split("-")[-1],
                append_label=args.append_label,
                args=args,
            )
        else:
            raise ValueError(f"Invalid model: {model_type}")

    print(model)
    return model


def build_link_prediction_model(model_type, n_layers, hidden_size, loss_fn_name, lr, data, args):
    assert n_layers >= 1, n_layers

    if "topdown" in model_type.lower():
        layer_sizes = [data.num_features] + [hidden_size] * n_layers + [0]

        model = GNNTowDownForward(
            layer_sizes=layer_sizes,
            optimizer_name='Adam',
            optimizer_kwargs={'lr': lr},
            loss_fn_name=loss_fn_name,
            gnn=model_type.split("-")[-1],
            args=args,
        )
    elif model_type.startswith("GNN_FL"):
        layer_sizes = [data.num_features] + [hidden_size] * n_layers

        if "FF".lower() in model_type.lower():
            forward_learning_type = "FF"
        else:
            forward_learning_type = "FL"

        model = GNNForward(
            layer_sizes=layer_sizes,
            optimizer_name='Adam',
            optimizer_kwargs={'lr': lr, 'weight_decay': 5e-4},
            forward_learning_type=forward_learning_type,
            gnn=model_type.split("-")[-1],
            args=args,
        )
    else:
        raise ValueError(f"Invalid model: {model_type}")

    print(model)
    return model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default="node-class", choices=["link-pred", "node-class"],
                        help="graph learning task")
    parser.add_argument('--model', type=str,
                        help="model type")
    parser.add_argument('--loss-fn-name', type=str, default="forwardforward_loss_fn",
                        help="loss function for forward training")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of gnn layers")
    parser.add_argument("--num-hidden", type=int, default=128,
                        help="number of hidden channels")
    parser.add_argument("--ff-theta", type=float, default=2.0,
                        help="theta for forward-forward training")
    parser.add_argument('--append-label', type=str, default="none",
                        choices=["none", "all", "input"])
    parser.add_argument('--dataset', type=str,
                        help="dataset name")
    parser.add_argument("--val-from", type=int, default=0,
                        help="epoch to start validation")
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
    parser.add_argument('--exp-setting', type=str, default="default",
                        help="experiment setting")
    parser.add_argument("--num-negs", type=int, default=2,
                        help="number of negative samples per postive")
    parser.add_argument('--overwrite-result', action='store_true')
    parser.set_defaults(overwrite_result=False)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="softmax temperature")
    parser.add_argument("--grad-max-norm", type=float, default=1.0,
                        help="max norm of the gradients")
    parser.add_argument('--aug-edge-direction', type=str, default="bidirection",
                        choices=["unidirection", "bidirection"])
    parser.add_argument("--test-time-steps", type=int, default=10)
    parser.add_argument("--storable-time-steps", type=str, default=None)
    parser.add_argument('--alternating-update', action='store_true')
    parser.set_defaults(alternating_update=False)
    args = parser.parse_args()
    return args


def populate_args(args):
    if args.append_label.lower() == "none":
        args.append_label = None

    setup_cuda(args)

    args.results_dir = settings.RESULTS_ROOT / args.exp_setting / args.dataset / args.task
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.exp_datetime = datetime.today().strftime('%Y%m%d_%H%M%S')
    stdout_path = args.results_dir / "stdout"
    stdout_path.mkdir(parents=True, exist_ok=True)
    log_stdout(stdout_path / f"stdout-{args.task}-{args.model}-{args.dataset}-{args.exp_datetime}.txt")

    storable_time_steps: Optional[str] = args.storable_time_steps
    if storable_time_steps is None:
        storable_time_steps = list(range(args.test_time_steps))[2:]
    else:
        try:
            storable_time_steps = list(range(int(storable_time_steps)))
        except Exception:
            storable_time_steps = list(map(int, storable_time_steps.strip().split(",")))
    assert all(t < args.test_time_steps for t in storable_time_steps), (args.test_time_steps, storable_time_steps)
    args.storable_time_steps = storable_time_steps

    from pprint import pformat
    print(f"args:\n{pformat(args.__dict__)}")
    return args


if __name__ == '__main__':
    args = parse_args()
    args = populate_args(args)
    main(args)
