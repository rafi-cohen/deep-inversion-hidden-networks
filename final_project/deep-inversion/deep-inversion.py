import torch
from torch import optim
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from matplotlib import pyplot
from PIL import Image, ImageFilter, ImageChops

CUDA_ENABLED = torch.cuda.is_available()
HYPERPARAMS = dict(lr=0.05,
                   batch_size=64,
                   a_tv=1e-4,
                   a_l2=0,
                   a_f=1e-2,
                   a_c=0.2)


class DeepInvert:
    def __init__(self, image, cuda):
        self.image = image
        self.cuda = cuda
        self.model = models.resnet18(pretrained=True)
        if cuda:
            self.model = self.model.cuda()
        # freeze model weights
        for p in self.model.parameters():
            p.requires_grad = False
        self.loss_fn = nn.CrossEntropyLoss()
        # resnet18 use 224x224 images
        imgSize = 224
        # imagenet normalization
        self.transformMean = [0.485, 0.456, 0.406]
        self.transformStd = [0.229, 0.224, 0.225]
        self.transformNormalise = transforms.Normalize(
            mean=self.transformMean,
            std=self.transformStd
        )

        self.transformPreprocess = transforms.Compose([
            transforms.Resize((imgSize, imgSize)),
            transforms.ToTensor(),
            self.transformNormalise
        ])

        self.tensorMean = torch.Tensor(self.transformMean)
        if cuda:
            self.tensorMean = self.tensorMean.cuda()

        self.tensorStd = torch.Tensor(self.transformStd)
        if cuda:
            self.tensorStd = self.tensorStd.cuda()

    def toImage(self, input):
        return input * self.tensorStd + self.tensorMean

    def deepInvert(self, image, iterations, target_onehot, lr):
        transformed = self.transformPreprocess(image).unsqueeze(0)
        if self.cuda:
            transformed = transformed.cuda()
        input = torch.autograd.Variable(transformed, requires_grad=True)
        # initialize the optimizer and register the image as a parameter
        optimizer = optim.Adam(input, lr)
        for _ in range(iterations):
            output = self.model(input)
            optimizer.zero_grad()
            loss = self.loss_fn(output, target_onehot)
            # R_DI = R_prior + R_feature
            # R_ADI = R_DI + R_compete
            # loss += R_ADI
            loss.backward()
            optimizer.step()

        input = input.data.squeeze()
        input.transpose_(0, 1)
        input.transpose_(1, 2)
        input = np.clip(self.toImage(input), 0, 1)
        return Image.fromarray(np.uint8(input * 255))
