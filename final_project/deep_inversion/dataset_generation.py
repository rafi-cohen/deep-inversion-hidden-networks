import imghdr
import os

import numpy as np
import torch

from deep_invert import DeepInvert
from parsing import parse_args

DATASET_SIZE = 5000
PARAMS = dict(a_f=1e-2,
              a_tv=8e-3,
              a_l2=1e-5,
              jitter=30,
              flip=0.5,
              lr=0.2,
              reg_fn='DI',
              targets=[1, 92, 107, 207, 270, 277, 294, 440, 920, 933],
              batch_size=100,
              iterations=20000,
              early_stopping=12,
              scheduler_patience=5,
              dataset='ImageNet',
              model_name='ResNet50',
              amp_mode='O2',
              output_dir='dataset')


def is_image(filename):
    return imghdr.what(filename) is not None


def rearrange_images_in_subfolders(dataset_dir):
    """
    Sorts images in a dataset folder into class-specific sub-folders (as required by torch.datasets.ImageFolder).
    Expected image filename format: 'batch_i_image_j_label_k.png'
    :param dataset_dir: the dataset's folder path
    """
    for image_filename in os.listdir(dataset_dir):
        if not is_image(image_filename):
            continue
        image_label = image_filename.replace('_', '.').split('.')[5]
        label_dir_path = os.path.join(dataset_dir, image_label)
        if not os.path.isdir(label_dir_path):
            os.mkdir(label_dir_path)
        original_image_path = os.path.join(dataset_dir, image_filename)
        new_image_path = os.path.join(label_dir_path, image_filename)
        os.replace(original_image_path, new_image_path)


def generate_dataset(params):
    targets_count = len(params['targets'])
    batch_size = params['batch_size']
    assert batch_size % targets_count == 0, 'batch size is not divisible by targets count'
    images_per_target = batch_size / targets_count
    batch_count = DATASET_SIZE // batch_size
    for i in range(batch_count):
        params['seed'] = i
        args = parse_args(params, add_timestamp=False)
        args.targets = torch.tensor(np.repeat(params['targets'], images_per_target))
        if args.cuda:
            args.targets = args.targets.cuda()
        DI = DeepInvert(**vars(args))
        images = DI.deepInvert(**vars(args))
        for j, image in enumerate(images):
            label = args.targets[j].item()
            image.save(os.path.join(args.output_dir, f'batch_{i}_image_{j}_label_{label}.png'))


def main():
    generate_dataset(PARAMS)
    rearrange_images_in_subfolders(PARAMS['output_dir'])


if __name__ == '__main__':
    main()
