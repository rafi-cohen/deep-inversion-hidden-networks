from itertools import product
from math import inf
from os import path
from pprint import pprint
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
import os

from inception_score.inception_score import inception_score
from parsing import parse_args
from params import MEANS, STDS
from deep_inversion import DeepInvert


GRID = dict(a_f=list(np.logspace(start=-8, stop=0, num=9)),
            a_tv=list(np.logspace(start=-8, stop=0, num=9)),
            a_l2=[0, 1e-2],
            lr=[0.05, 0.005],
            reg_fn=['DI'],
            target=[294],
            batch_size=[50],
            iterations=[20000])


def dict_product(dictionary):
    return (dict(zip(dictionary.keys(), values)) for values in product(*dictionary.values()))


def underscore_to_dash(string):
    return string.replace('_', '-')


def grid_search(grid):
    best_score = -inf
    best_configuration = None
    preprocess = Compose([ToTensor(), Normalize(MEANS['ImageNet'], STDS['ImageNet'])])
    for configuration in dict_product(grid):
        args = []
        for key, value in configuration.items():
            args.extend(f'--{underscore_to_dash(key)} {value}'.split())

        args = parse_args(args)

        DI = DeepInvert(**vars(args))
        images = DI.deepInvert(**vars(args))
        for i, image in enumerate(images):
            image.save(os.path.join(args.output_dir, f'{i}.png'))
        images = [preprocess(image) for image in images]
        images_dataset = torch.stack(images)
        score = inception_score(images_dataset, resize=True, batch_size=1)[0]

        with open(path.join(args.output_dir, 'inception_score.txt'), 'w') as f:
            print(score, file=f)

        if score > best_score:
            best_score = score
            best_configuration = configuration

        print(f'Best score is {best_score}, with the configuration:')
        pprint(best_configuration)


if __name__ == '__main__':
    grid_search(GRID)
