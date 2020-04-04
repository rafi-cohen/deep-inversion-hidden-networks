import argparse
import random
import torch
import torch.nn as nn
from os import path
from PIL import Image
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


CUDA_ENABLED = torch.cuda.is_available()
if CUDA_ENABLED:
    torch.cuda.empty_cache()
# vgg11_bn
# resnet18
model_name = 'resnet18'
VGG11_BN_PARAMS = dict(model_name='vgg11_bn',
                       image_size=(32, 32),
                       mean=[0.4914, 0.4822, 0.4465],
                       std=[0.2023, 0.1994, 0.2010],
                       batch_size=30,
                       target_class=8,
                       lr=0.05,
                       a_tv=2.5e-5,
                       a_l2=3e-8,
                       a_f=1,
                       a_c=0.2)
RESNET34_PARAMS = dict(model_name='resnet34',
                       image_size=(32, 32),
                       mean=[0.4914, 0.4822, 0.4465],
                       std=[0.2023, 0.1994, 0.2010],
                       batch_size=30,
                       target_class=5,
                       lr=0.05,
                       a_tv=2.5e-5,
                       a_l2=3e-8,
                       a_f=1,
                       a_c=0.2)
RESNET18_PARAMS = dict(model_name='resnet18',
                       image_size=(224, 224),
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       batch_size=10,
                       target_class=950,
                       lr=0.05,
                       a_tv=1e-6,
                       a_l2=0,
                       a_f=1e-2,
                       a_c=0.2)

PARAMS_DICT = dict(vgg11_bn=VGG11_BN_PARAMS,
                   resnet34=RESNET34_PARAMS,
                   resnet18=RESNET18_PARAMS)

HYPERPARAMS = PARAMS_DICT[model_name]

OUT_DIR = 'generated'
FILENAME_FORMAT = path.join(OUT_DIR, '{}.jpg')


def main():
    # Load a local image:
    # batch = Image.open(path.join(OUT_DIR, '9.jpg'))
    # preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    # batch = preprocess(batch).unsqueeze(0)

    # Create random noise:
    batch = torch.randn(HYPERPARAMS['batch_size'], 3, *HYPERPARAMS['image_size'])

    target_criterion = HYPERPARAMS['target_class'] * torch.ones(HYPERPARAMS['batch_size'], dtype=torch.long)

    if HYPERPARAMS['model_name'] == 'vgg11_bn':
        model = models.vgg11_bn(pretrained=True)
    elif HYPERPARAMS['model_name'] == 'resnet34':
        model = models.resnet34(pretrained=True)
    else:
        model = models.resnet18(pretrained=True)

    if CUDA_ENABLED:
        model = model.cuda()
        batch = batch.cuda()
        target_criterion = target_criterion.cuda()
    DI = DeepInvert(model, HYPERPARAMS['mean'], HYPERPARAMS['std'], CUDA_ENABLED,
                    HYPERPARAMS['a_tv'], HYPERPARAMS['a_l2'], HYPERPARAMS['a_f'])
    images = DI.deepInvert(batch, iterations=2000,
                           target=target_criterion, lr=HYPERPARAMS['lr'])
    for i, image in enumerate(images):
        image.save(FILENAME_FORMAT.format(i))


if __name__ == '__main__':
    main()
