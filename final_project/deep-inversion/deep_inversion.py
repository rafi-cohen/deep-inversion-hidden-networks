import torch
from torch import optim
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

reg_feature = torch.tensor([0.0], requires_grad=True)
if torch.cuda.is_available():
    reg_feature = reg_feature.cuda()


class DeepInvert:
    def __init__(self, model, mean, std, cuda, loss_fn, reg_fn, a_f, *args, **kwargs):
        self.transformMean = mean
        self.transformStd = std

        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.loss_fn = loss_fn
        self.reg_fn = reg_fn
        self.a_f = a_f
        self.transformPreprocess = transforms.Normalize(mean=self.transformMean, std=self.transformStd)
        self.tensorMean = torch.Tensor(self.transformMean)
        self.tensorStd = torch.Tensor(self.transformStd)
        if cuda:
            self.tensorMean = self.tensorMean.cuda()
            self.tensorStd = self.tensorStd.cuda()
        self.cuda = cuda
        self.handles = []

    def clip(self, image_tensor):
        for c in range(3):
            m, s = self.transformMean[c], self.transformStd[c]
            image_tensor[0, c] = torch.clamp(image_tensor[0, c], -m / s, (1 - m) / s)
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

    @staticmethod
    def _hook(module, input, output):
        if isinstance(module, nn.BatchNorm2d):
            global reg_feature
            current_feature_map = input[0]
            feature_map_mean = torch.mean(current_feature_map, dim=(0, 2, 3))
            feature_map_var = torch.var(current_feature_map, dim=(0, 2, 3), unbiased=False)
            reg_feature = reg_feature + torch.norm(feature_map_mean - module.running_mean)
            reg_feature = reg_feature + torch.norm(feature_map_var - module.running_var)

    def _register_hooks(self):
        self.handles = []
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                handle = module.register_forward_hook(self._hook)
                self.handles.append(handle)

    def _remove_hooks(self):
        for handle in self.handles:
            handle.remove()

    def deepInvert(self, batch, iterations, target, lr, *args, **kwargs):
        global reg_feature
        transformed_images = []
        for image in batch:
            transformed_images.append(self.transformPreprocess(image))
        input = torch.stack(transformed_images)

        input.requires_grad_(True)
        if self.cuda:
            input = input.cuda()
        # initialize the optimizer and register the image as a parameter
        optimizer = optim.Adam([input], lr)
        self._register_hooks()
        with tqdm(total=iterations) as pbar:
            for i in range(iterations):
                output = self.model(input)
                optimizer.zero_grad()
                reg_feature.zero_()
                reg_feature.detach_()
                ce_loss = self.loss_fn(output, target)
                loss = ce_loss
                tv_reg, l2_reg = self.reg_fn(input)
                loss = loss + tv_reg + l2_reg
                loss = loss + self.a_f * reg_feature
                loss.backward()
                optimizer.step()
                # clip the image after every gradient step
                input.data = self.clip(input.data)

                desc_str = f'#{i}: total_loss = {loss.item()}'
                pbar.set_description(desc_str)
                pbar.update()
        self._remove_hooks()

        return self.toImages(input)
