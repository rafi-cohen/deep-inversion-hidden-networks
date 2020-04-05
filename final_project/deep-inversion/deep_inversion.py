import torch
from torch import optim
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm


class DeepInvert:
    def __init__(self, model, mean, std, cuda, loss_fn, reg_fn, *args, **kwargs):
        self.transformMean = mean
        self.transformStd = std

        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.loss_fn = loss_fn
        self.reg_fn = reg_fn
        self.transformPreprocess = transforms.Normalize(mean=self.transformMean, std=self.transformStd)
        self.tensorMean = torch.Tensor(self.transformMean)
        self.tensorStd = torch.Tensor(self.transformStd)
        if cuda:
            self.tensorMean = self.tensorMean.cuda()
            self.tensorStd = self.tensorStd.cuda()
        self.cuda = cuda

    def clip(self, image_tensor):
        for c in range(3):
            m, s = self.transformMean[c], self.transformStd[c]
            image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
        return image_tensor

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

    def deepInvert(self, batch, iterations, target, lr, *args, **kwargs):
        transformed_images = []
        for image in batch:
            transformed_images.append(self.transformPreprocess(image))
        input = torch.stack(transformed_images)

        input.requires_grad_(True)
        if self.cuda:
            input = input.cuda()
        # initialize the optimizer and register the image as a parameter
        optimizer = optim.Adam([input], lr)
        with tqdm(total=iterations) as pbar:
            for i in range(iterations):
                output = self.model(input)
                optimizer.zero_grad()
                loss = self.loss_fn(output, target) + self.reg_fn(input)
                loss.backward()
                optimizer.step()
                # clip the image after every gradient step
                input.data = self.clip(input.data)

                desc_str = f'#{i}: total_loss = {loss.item()}'
                pbar.set_description(desc_str)
                pbar.update()

        return self.toImages(input)
