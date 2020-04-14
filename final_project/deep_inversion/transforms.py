import torch
import torch.nn as nn
import torchvision.transforms.functional as F


class Denormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
        self.demean = [-m/s for m, s in zip(mean, std)]
        self.destd = [1/s for s in std]

    def __call__(self, tensor):
        return F.normalize(tensor, self.demean, self.destd, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
