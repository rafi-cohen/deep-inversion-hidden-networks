import argparse
import random
import torch
import torch.nn as nn
import os
from datetime import datetime
from pprint import pprint

from params import (MEANS,
                    STDS,
                    DIMS,
                    DATASETS,
                    MODELS,
                    MODEL_NAMES,
                    REGULARIZATIONS,
                    REG_FNS,
                    AMP_MODES
                    )


def create_parser():
    parser = argparse.ArgumentParser(description="Deep Inversion")

    parser.add_argument("--iterations", type=int, default=160, metavar="I",
                        help="number of epochs to train (default: 10)")

    parser.add_argument("--batch-size", type=int, default=128, metavar="B",
                        help="input batch size for training (default: 128)")

    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training")

    parser.add_argument("--amp-mode", type=str, default="off", metavar="AMP", choices=AMP_MODES,
                        help="Automatic Mixed Precision mode (default: off)")

    parser.add_argument("--seed", type=int, default=None, metavar="S",
                        help="random seed (default: None)")

    parser.add_argument("--lr", type=float, default=0.05, metavar="LR",
                        help="learning rate (default: 0.05)")

    parser.add_argument("--target", type=int, default=294, metavar="T",
                        help="target class for image synthesis (default: 294)")

    parser.add_argument("--data-set", type=str, default="ImageNet", metavar="DS", choices=DATASETS,
                        help="Dataset to perform synthesis on (default: ImageNet)")

    parser.add_argument("--model-name", type=str, default="ResNet50", metavar="MN", choices=MODEL_NAMES,
                        help="Name of model to use for synthesis (default: ResNet50)")

    parser.add_argument("--jitter", type=int, default=0, metavar="J",
                        help="Amount of jitter to apply on each iteration (default: 0)")

    parser.add_argument("--reg-fn", type=str, default="prior", metavar="RF",
                        choices=REG_FNS, help="Regularization function (default: prior)")

    parser.add_argument("--a-tv", type=float, default=1e-4, metavar="ATV",
                        help="TV regularization factor (default: 1e-4)")

    parser.add_argument("--a-l2", type=float, default=0, metavar="AL2",
                        help="l2-Norm regularization factor (default: 0)")

    parser.add_argument("--a-f", type=float, default=1e-2, metavar="AF",
                        help="Feature regularization factor (default: 1e-2)")

    parser.add_argument("--a-c", type=float, default=0.2, metavar="AC",
                        help="Compete regularization factor (default: 0.2)")

    parser.add_argument("--output-dir", type=str, default="generated", metavar="OD",
                        help="Directory for storing generated images (default: generated)")
    return parser


def parse_args(args=None):
    parser = create_parser()
    args = parser.parse_args(args)

    args.output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    assert os.path.isdir(args.output_dir), 'Could not create output directory'

    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        pprint(vars(args), stream=f)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        random.seed(torch.initial_seed())

    args.model = MODELS[args.data_set][args.model_name](pretrained=True)
    args.batch = torch.randn(args.batch_size, *DIMS[args.data_set])
    args.target = torch.empty(args.batch_size, dtype=torch.long).fill_(args.target)
    if args.cuda:
        args.model = args.model.cuda()
        args.batch = args.batch.cuda()
        args.target = args.target.cuda()
    args.mean = MEANS[args.data_set]
    args.std = STDS[args.data_set]
    args.reg_fn = REGULARIZATIONS[args.reg_fn]
    if args.reg_fn:
        # instantiate reg_fn if it is not None
        args.reg_fn = args.reg_fn(**vars(args))
    args.loss_fn = nn.CrossEntropyLoss()
    return args
