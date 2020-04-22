import os
from itertools import product
from math import inf
from os import path
from pprint import pprint

import torch
from torchvision.transforms import Compose, ToTensor, Normalize

from deep_invert import DeepInvert
from inception_score.inception_score import inception_score
from params import MEANS, STDS, LABELS
from parsing import parse_args

GRID = dict(a_f=[5e-3, 1e-2],
            a_tv=[5e-4, 5e-3, 1e-3],
            a_l2=[0],
            jitter=[30],
            flip=[0.5],
            lr=[0.2, 0.05],
            reg_fn=['DI'],
            targets=[[294, 335, 985, 968, 354, 113, 572, 366, 701, 749,
                      779, 928, 953, 954, 971, 980, 440, 486, 76, 130]],
            batch_size=[50],
            iterations=[20000],
            early_stopping=[25],
            scheduler_patience=[5],
            dataset=['ImageNet'],
            model_name=['ResNet50'],
            amp_mode=['O2'],
            seed=[42])


def dict_product(dictionary):
    return (dict(zip(dictionary.keys(), values)) for values in product(*dictionary.values()))


def grid_search(grid):
    best_score = -inf
    best_configuration = None
    dataset = grid['dataset'][0]
    preprocess = Compose([ToTensor(), Normalize(MEANS[dataset], STDS[dataset])])
    for configuration in dict_product(grid):
        args = parse_args(configuration)

        DI = DeepInvert(**vars(args))
        images = DI.deepInvert(**vars(args))
        for i, image in enumerate(images):
            label = LABELS[args.dataset][args.targets[i].item()]
            image.save(os.path.join(args.output_dir, f'{i}_{label}.png'))
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
