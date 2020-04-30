import os

import torch
import torch.nn as nn
from torchvision import transforms

from deep_invert import DeepInvert
from params import MODELS, LABELS
from parsing import parse_args


def main():
    args = parse_args()

    DI = DeepInvert(**vars(args))
    images = DI.deepInvert(**vars(args))
    for i, image in enumerate(images):
        label = LABELS[args.dataset][args.targets[i].item()]
        image.save(os.path.join(args.output_dir, f'{i}_{label}.png'))


if __name__ == '__main__':
    main()
