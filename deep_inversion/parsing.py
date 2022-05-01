import argparse
import os
import random
from datetime import datetime
from pprint import pprint

import torch

from params import (MEANS,
                    STDS,
                    DIMS,
                    DATASETS,
                    MODELS,
                    MODEL_NAMES,
                    LOSSES,
                    LOSS_FNS,
                    REGULARIZATIONS,
                    REG_FNS,
                    AMP_MODES,
                    DATASETS_CLASS_COUNT)


def create_parser():
    parser = argparse.ArgumentParser(description="Deep Inversion")

    parser.add_argument("--iterations", type=int, default=20000, metavar="I",
                        help="number of epochs to run Deep Inversion for (default: 20000)")

    parser.add_argument("--early-stopping", type=int, default=15, metavar="ES",
                        help="percentage of iterations with no improvement"
                             " to wait before early stopping (default: 15)")

    parser.add_argument("--batch-size", type=int, default=128, metavar="B",
                        help="number of images to generate in a batch (default: 128)")

    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disable CUDA")

    parser.add_argument("--amp-mode", type=str, default="O2", metavar="AMP", choices=AMP_MODES,
                        help="Automatic Mixed Precision mode (default: O2)")

    parser.add_argument("--seed", type=int, default=None, metavar="S",
                        help="random seed (default: None)")

    parser.add_argument("--lr", type=float, default=0.2, metavar="LR",
                        help="learning rate (default: 0.2)")

    parser.add_argument("--scheduler-patience", type=int, default=5, metavar="SP",
                        help="learning rate scheduler patience in percentage"
                             " relative to the number of iterations (default: 5)")

    parser.add_argument("--targets", type=int, nargs='+', default=[-1], metavar="T",
                        help="target classes for image synthesis, or -1 for randomization (default: -1)")

    parser.add_argument("--dataset", type=str, default="ImageNet", metavar="DS", choices=DATASETS,
                        help="dataset to perform synthesis on (default: ImageNet)")

    parser.add_argument("--model-name", type=str, default="ResNet50", metavar="MN", choices=MODEL_NAMES,
                        help="name of model to use for synthesis (default: ResNet50)")

    parser.add_argument("--jitter", type=int, default=30, metavar="J",
                        help="amount of jitter to apply on each iteration (default: 30)")

    parser.add_argument("--flip", type=float, default=0.5, metavar="FLP",
                        help="horizontal flip probability (default: 0.5)")

    parser.add_argument("--loss-fn", type=str, default="CE", metavar="LF",
                        choices=LOSS_FNS, help="loss function (default: CE)")

    parser.add_argument("--temp", type=float, default=1, metavar="TMP",
                        help="temperature value for CrossEntropyLoss (default: 1)")

    parser.add_argument("--reg-fn", type=str, default="DI", metavar="RF",
                        choices=REG_FNS, help="regularization function (default: DI)")

    parser.add_argument("--a-tv", type=float, default=8e-3, metavar="ATV",
                        help="TV regularization factor (default: 8e-3)")

    parser.add_argument("--a-l2", type=float, default=1e-5, metavar="AL2",
                        help="l2-norm regularization factor (default: 1e-5)")

    parser.add_argument("--a-f", type=float, default=1e-2, metavar="AF",
                        help="feature regularization factor (default: 1e-2)")

    parser.add_argument("--output-dir", type=str, default="generated", metavar="OD",
                        help="directory for storing generated images (default: generated)")
    return parser


def underscore_to_dash(string):
    return string.replace('_', '-')


def dict_to_args(dict_args):
    args = []
    for key, value in dict_args.items():
        if isinstance(value, list):
            value = ' '.join(map(str, value))
        args.extend(f'--{underscore_to_dash(key)} {value}'.split())
    return args


def parse_args(args=None, add_timestamp=True):
    parser = create_parser()
    if isinstance(args, dict):
        args = dict_to_args(args)
    args = parser.parse_args(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        random.seed(torch.initial_seed())

    if args.targets[0] == -1:
        # randomize class targets from the entire dataset
        args.targets = torch.randint(low=0, high=DATASETS_CLASS_COUNT[args.dataset], size=(args.batch_size,),
                                     dtype=torch.long)
    else:
        # randomize class targets from the user's predefined list
        targets = random.choices(args.targets, k=args.batch_size)
        args.targets = torch.tensor(targets, dtype=torch.long)

    if add_timestamp:
        args.output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    assert os.path.isdir(args.output_dir), 'Could not create output directory'

    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        pprint(vars(args), stream=f)

    args.model = MODELS[args.dataset][args.model_name](pretrained=True)
    args.batch = torch.randn(args.batch_size, *DIMS[args.dataset])
    if args.cuda:
        args.model = args.model.cuda()
        args.batch = args.batch.cuda()
        args.targets = args.targets.cuda()
        torch.backends.cudnn.benchmark = True
    args.mean = MEANS[args.dataset]
    args.std = STDS[args.dataset]
    if args.temp != 1:
        args.loss_fn = LOSSES[args.loss_fn](args.temp)
    else:
        args.loss_fn = LOSSES[args.loss_fn]()
    args.reg_fn = REGULARIZATIONS[args.reg_fn]
    if args.reg_fn:
        # instantiate reg_fn if it is not None
        args.reg_fn = args.reg_fn(**vars(args))
    return args
