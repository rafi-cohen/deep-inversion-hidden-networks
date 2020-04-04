import argparse
import random
import torch
import torch.nn as nn
import os
from torchvision import transforms, models

import cifar10_models
from deep_inversion import DeepInvert
from utils import PriorRegularization, DIRegularization

MEANS = dict(ImageNet=[0.485, 0.456, 0.406], CIFAR10=[0.4914, 0.4822, 0.4465])
STDS = dict(ImageNet=[0.229, 0.224, 0.225], CIFAR10=[0.2023, 0.1994, 0.2010])
DIMS = dict(ImageNet=(3, 224, 224), CIFAR10=(3, 32, 32))
DATASETS = list(MEANS.keys())

IMAGE_NET_MODELS = dict(ResNet18=models.resnet18,
                        ResNet34=models.resnet34,
                        ResNet50=models.resnet50,
                        vgg11=models.vgg11_bn,
                        vgg13=models.vgg13_bn,
                        vgg16=models.vgg16_bn,
                        vgg19=models.vgg19_bn)
CIFAR10_MODELS = dict(ResNet18=cifar10_models.resnet18,
                      ResNet34=cifar10_models.resnet34,
                      ResNet50=cifar10_models.resnet50,
                      vgg11=cifar10_models.vgg11_bn,
                      vgg13=cifar10_models.vgg13_bn,
                      vgg16=cifar10_models.vgg16_bn,
                      vgg19=cifar10_models.vgg19_bn)
MODELS = dict(ImageNet=IMAGE_NET_MODELS, CIFAR10=CIFAR10_MODELS)
MODEL_NAMES = list(IMAGE_NET_MODELS.keys())

REGULARIZATIONS = dict(prior=PriorRegularization, DI=DIRegularization, none=None)
REG_FNS = list(REGULARIZATIONS.keys())


def create_parser():
    parser = argparse.ArgumentParser(description="Deep Inversion")

    parser.add_argument("--iterations", type=int, default=160, metavar="I",
                        help="number of epochs to train (default: 10)")

    parser.add_argument("--batch-size", type=int, default=128, metavar="B",
                        help="input batch size for training (default: 128)")

    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training")

    parser.add_argument("--seed", type=int, default=None, metavar="S",
                        help="random seed (default: None)")

    parser.add_argument("--lr", type=float, default=0.05, metavar="LR",
                        help="learning rate (default: 0.05)")

    parser.add_argument("--target", type=int, default=294, metavar="T",
                        help="target class for image synthesis (default: 294)")

    parser.add_argument("--data-set", type=str, default="ImageNet", metavar="DS", choices=DATASETS,
                        help="Dataset to perform synthesis on (default: ImageNet)")

    parser.add_argument("--model-name", type=str, default="ResNet18", metavar="MN", choices=MODEL_NAMES,
                        help="Name of model to use for synthesis (default: ResNet18)")

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


def parse_args():
    parser = create_parser()
    args = parser.parse_args()

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


def evaluate(images, model, data_set, model_name, mean, std, cuda, *args, **kwargs):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    softmax = nn.Softmax(dim=1)
    original_model = MODELS[data_set][model_name](pretrained=True)
    if cuda:
        original_model.cuda()
    for i, image in enumerate(images):
        image = preprocess(image).unsqueeze(0)
        if cuda:
            image = image.cuda()
        original_confidence, original_pred = torch.max(softmax(original_model(image)), dim=1)
        new_confidence, new_pred = torch.max(softmax(model(image)), dim=1)
        print(f'image #{i}')
        print(f'original_pred = {original_pred.item()}, original_confidence = {original_confidence.item()}')
        print(f'new_pred = {new_pred.item()}, new_confidence = {new_confidence.item()}')
        # verify that the model was not changed during training (i.e. the results are identical)
        assert original_pred == new_pred
        assert original_confidence == new_confidence


def main():
    args = parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    assert os.path.isdir(args.output_dir), 'Could not create output directory'

    DI = DeepInvert(**vars(args))
    images = DI.deepInvert(**vars(args))
    for i, image in enumerate(images):
        image.save(os.path.join(args.output_dir, f'{i}.jpg'))
    evaluate(images, **vars(args))


if __name__ == '__main__':
    main()
