import torch
from os import path
from PIL import Image
from deep_inversion import DeepInvert

CUDA_ENABLED = torch.cuda.is_available()
HYPERPARAMS = dict(batch_size=2,
                   target_class=957,
                   lr=0.05,
                   a_tv=1e-4,
                   a_l2=0,
                   a_f=1e-2,
                   a_c=0.2)
OUT_DIR = 'generated'
FILENAME_FORMAT = path.join(OUT_DIR, '{}.jpg')


def main():
    batch = torch.randn(HYPERPARAMS['batch_size'], 3, 224, 224)
    target_onehot = torch.zeros(1000)
    target_onehot[HYPERPARAMS['target_class']] = 1
    DI = DeepInvert(batch, CUDA_ENABLED,
                    HYPERPARAMS['a_tv'], HYPERPARAMS['a_l2'], HYPERPARAMS['a_f'])
    images = DI.deepInvert(batch, iterations=2000,
                           target_onehot=target_onehot, lr=HYPERPARAMS['lr'])
    for i, image in enumerate(images):
        image.save(FILENAME_FORMAT.format(i))
