import torch
from torch import optim
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import DIRegularization
from cifar10_models import *

reg_feature = torch.tensor([0.0], requires_grad=True)
if torch.cuda.is_available():
    reg_feature = reg_feature.cuda()

class DeepInvert:
    def __init__(self, model_name, model_mean, model_std, cuda, a_tv, a_l2, a_f):
        self.transformMean = model_mean
        self.transformStd = model_std

        if model_name == 'vgg11_bn':
            self.model = vgg11_bn(pretrained=True)
        elif model_name == 'resnet34':
            self.model = resnet34(pretrained=True)
        else:
            self.model = models.resnet18(pretrained=True)

        # self.model.eval()

        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

        self.loss_fn = nn.CrossEntropyLoss()
        self.a_f = a_f
        self.reg_fn = DIRegularization(self.model, a_tv, a_l2, a_f)
        self.transformPreprocess = transforms.Normalize(mean=self.transformMean, std=self.transformStd)
        self.tensorMean = torch.Tensor(self.transformMean)
        self.tensorStd = torch.Tensor(self.transformStd)
        if cuda:
            self.tensorMean = self.tensorMean.cuda()
            self.tensorStd = self.tensorStd.cuda()
            self.model = self.model.cuda()
        self.cuda = cuda

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

    def forward(self, input):

        def hook(module, input, output):
            if isinstance(module, nn.BatchNorm2d):
                global reg_feature
                current_feature_map = input[0].transpose(0, 1)
                feature_map_mean = torch.mean(current_feature_map, dim=(1, 2, 3))
                feature_map_var = torch.var(current_feature_map, dim=(1, 2, 3), unbiased=False)
                reg_feature = reg_feature + torch.norm(feature_map_mean - module.running_mean)
                reg_feature = reg_feature + torch.norm(feature_map_var - module.running_var)

        handles = []
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                handle = module.register_forward_hook(hook)
                handles.append(handle)
        handles.append(list(self.model.modules())[-4].register_forward_hook(hook))
        output = self.model(input)

        for handle in handles:
            handle.remove()

        return output

    def deepInvert(self, batch, iterations, target_criterion, lr):
        transformed_images = []
        for image in batch:
            transformed_images.append(self.transformPreprocess(image))
        input = torch.stack(transformed_images)

        #sanity check:
        # input_class = nn.Softmax()(self.model(transformed)).argmax()

        input.requires_grad_(True)
        if self.cuda:
            input = input.cuda()
        # initialize the optimizer and register the image as a parameter
        optimizer = optim.Adam([input], lr)
        output = []
        with tqdm(total=iterations) as pbar:
            for i in range(iterations):
                output = self.forward(input)
                optimizer.zero_grad()
                ce_loss = self.loss_fn(output, target_criterion)
                loss = ce_loss
                tv_reg, l2_reg = self.reg_fn(input)
                loss = loss + tv_reg + l2_reg
                global reg_feature
                loss = loss + self.a_f * reg_feature
                loss.backward()
                optimizer.step()
                # clip the image after every gradient step
                input.data = self.clip(input.data)
                # random horizontal flipping
                # if torch.rand(1)[0] > 0.5:
                #     input.data = input.data.flip(3)
                desc_str = f'#{i}: total_loss = {loss.item()}'
                # desc_str = f'#{i}: total_loss = {loss.item()} loss_ce = {ce_loss.item()} r_feature =' \
                #            f' {self.a_f * reg_feature.item()} ' \
                #            f'r_tv = {tv_reg.item()} r_l2 = {l2_reg.item()} '
                pbar.set_description(desc_str)
                pbar.update()
                reg_feature.zero_()
                reg_feature.detach_()
                target_class_idx = target_criterion[0].item()
                # print(nn.Softmax()(output)[:, target_class_idx])
                # print(torch.argmax(nn.Softmax()(output), dim=1))

        return self.toImages(input)

# class AdaptiveDeepInvert(DeepInvert):
#     def __init__(self, image, cuda, a_tv, a_l2):
#         super.__init__(image, cuda, a_tv, a_l2)
#         self.reg_fn = ADIRegularization()
