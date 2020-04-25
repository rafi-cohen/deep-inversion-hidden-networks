import os
import numpy as np
import torch
from deep_invert import DeepInvert
from parsing import parse_args

DATASET_SIZE = 10000
PARAMS = dict(a_f=1e-2,
              a_tv=8e-3,
              a_l2=1e-5,
              jitter=30,
              flip=0.5,
              lr=0.2,
              reg_fn='DI',
              targets=[294, 1, 933, 980, 63, 92, 107, 985, 207, 270, 277, 283, 360, 968, 440,
                       417, 590, 762, 920, 574],
              batch_size=100,
              iterations=20000,
              early_stopping=12,
              scheduler_patience=5,
              dataset='ImageNet',
              model_name='ResNet50',
              amp_mode='O2',
              output_dir='generated')


def generate_dataset(params):
    targets_count = len(params['targets'])
    batch_size = params['batch_size']
    assert batch_size % targets_count == 0, 'batch size is not divisible by targets count'
    images_per_target = batch_size / targets_count
    batch_count = DATASET_SIZE // batch_size
    for i in range(batch_count):
        params['seed'] = i
        args = parse_args(params, create_output_directory=False)
        args.targets = torch.tensor(np.repeat(params['targets'], images_per_target))
        DI = DeepInvert(**vars(args))
        images = DI.deepInvert(**vars(args))
        for j, image in enumerate(images):
            label = args.targets[j].item()
            image.save(os.path.join(args.output_dir, f'batch_{i}_image_{j}_label_{label}.png'))


if __name__ == '__main__':
    generate_dataset(PARAMS)
