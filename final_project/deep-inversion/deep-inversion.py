import torch
from torch import optim
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
from utils import DIRegularization

CUDA_ENABLED = torch.cuda.is_available()
HYPERPARAMS = dict(lr=0.05,
                   batch_size=64,
                   a_tv=1e-4,
                   a_l2=0,
                   a_f=1e-2,
                   a_c=0.2)


class DeepInvert:
    def __init__(self, image, cuda, a_tv, a_l2, a_f):
        self.image = image
        self.cuda = cuda
        self.model = models.resnet18(pretrained=True)
        if cuda:
            self.model = self.model.cuda()
        # freeze model weights
        for p in self.model.parameters():
            p.requires_grad = False
        self.loss_fn = nn.CrossEntropyLoss()
        self.reg_fn = DIRegularization(self.model, a_tv, a_l2, a_f)
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

    def toImages(self, input):
        input.transpose_(1, 2)
        input.transpose_(2, 3)
        images = []
        for image in input:
            normalized = image * self.tensorStd + self.tensorMean
            clipped = np.clip(normalized, 0, 1)
            images.append(Image.fromarray(np.uint8(clipped * 255)))
        return images

    def deepInvert(self, batch, iterations, target_onehot, lr):
        transformed = self.transformPreprocess(batch)
        if self.cuda:
            transformed = transformed.cuda()
        input = torch.autograd.Variable(transformed, requires_grad=True)
        # initialize the optimizer and register the image as a parameter
        optimizer = optim.Adam(input, lr)
        for _ in range(iterations):
            output = input
            batch_running_means = []
            batch_running_vars = []
            for module in self.model.modules:
                output = module(output)
                if isinstance(module, nn.Conv2d):
                    batch_running_means.append(torch.mean(output, dim=0))
                    batch_running_vars.append(torch.var(output, dim=0))
            optimizer.zero_grad()
            loss = self.loss_fn(output, target_onehot)
            loss += self.reg_fn(output, batch_running_means,
                                batch_running_vars)
            # R_ADI = R_DI + R_compete
            loss.backward()
            optimizer.step()

        return self.toImages(output)

# class AdaptiveDeepInvert(DeepInvert):
#     def __init__(self, image, cuda, a_tv, a_l2):
#         super.__init__(image, cuda, a_tv, a_l2)
#         self.reg_fn = ADIRegularization()
