from torchvision import models

from cifar10_models import cifar10_models
from utils import PriorRegularization, DIRegularization

MEANS = dict(ImageNet=[0.485, 0.456, 0.406], CIFAR10=[0.4914, 0.4822, 0.4465])
STDS = dict(ImageNet=[0.229, 0.224, 0.225], CIFAR10=[0.2023, 0.1994, 0.2010])
DIMS = dict(ImageNet=(3, 224, 224), CIFAR10=(3, 32, 32))
DATASETS = list(MEANS.keys())
DATASETS_CLASS_COUNT = dict(ImageNet=1000, CIFAR10=10)

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
AMP_MODES = ['off', 'O0', 'O1', 'O2', 'O3']
