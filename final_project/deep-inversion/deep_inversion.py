import torch
from torch import optim
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import DIRegularization


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

        self.transformPreprocess = self.transformNormalise

        self.tensorMean = torch.Tensor(self.transformMean)
        if cuda:
            self.tensorMean = self.tensorMean.cuda()

        self.tensorStd = torch.Tensor(self.transformStd)
        if cuda:
            self.tensorStd = self.tensorStd.cuda()

    def toImages(self, input):
        images = []
        with torch.no_grad():
            input.transpose_(1, 2)
            input.transpose_(2, 3)
            for image in input:
                normalized = (image * self.tensorStd + self.tensorMean).cpu()
                clipped = np.clip(normalized, 0, 1)
                images.append(Image.fromarray(np.uint8(clipped * 255)))
        return images

    def forward(self, input):
        batch_running_means = []
        batch_running_vars = []

        def hook(module, input, output):
            if isinstance(module, nn.Conv2d):
                shape = output.shape
                batch_running_means.append(output.view(shape[0], shape[1], -1).mean(dim=2).mean(dim=0))
                if shape[0] == 1:
                    batch_running_vars.append(0)
                else:
                    batch_running_vars.append(output.view(shape[0], shape[1], -1).var(dim=2).var(dim=0))

        handles = []
        for module in self.model.modules():
            handle = module.register_forward_hook(hook)
            handles.append(handle)

        output = self.model(input)

        for handle in handles:
            handle.remove()

        return output, batch_running_means, batch_running_vars

    def deepInvert(self, batch, iterations, target_criterion, lr):
        transformed_images = []
        for image in batch:
            transformed_images.append(self.transformPreprocess(image))
        transformed = torch.stack(transformed_images)
        if self.cuda:
            transformed = transformed.cuda()
        input = torch.autograd.Variable(transformed, requires_grad=True)
        # initialize the optimizer and register the image as a parameter
        optimizer = optim.Adam([input], lr)
        output = []
        with tqdm(total=iterations) as pbar:
            for i in range(iterations):
                output, batch_running_means, batch_running_vars = self.forward(input)
                optimizer.zero_grad()
                loss = self.loss_fn(output, target_criterion)
                loss += self.reg_fn(input, batch_running_means,
                                    batch_running_vars)
                # R_ADI = R_DI + R_compete
                loss.backward()
                optimizer.step()

                desc_str = f'#{i}: loss = {loss}'
                pbar.set_description(desc_str)
                pbar.update()

        return self.toImages(input)

# class AdaptiveDeepInvert(DeepInvert):
#     def __init__(self, image, cuda, a_tv, a_l2):
#         super.__init__(image, cuda, a_tv, a_l2)
#         self.reg_fn = ADIRegularization()
