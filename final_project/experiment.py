import os
import sys
sys.path.extend(['deep_inversion', 'hidden_networks'])

from deep_inversion.deep_invert import DeepInvert
from deep_inversion.params import LABELS
from deep_inversion.parsing import parse_args as parse_DIargs
from hidden_networks.args import get_config
from hidden_networks.main import args as HNargs
from hidden_networks.main import get_model, pretrained


def GetPretrainedSubnet(config, weights):
    HNargs.config = config
    get_config(HNargs)
    HNargs.pretrained = weights
    HNargs.prune_rate = 0.5
    HNargs.multigpu = [0]
    model = get_model(HNargs)
    pretrained(HNargs, model)
    return model


def GetDIargs():
    DIargs = dict(a_f=5e-3,
                  a_tv=5e-4,
                  a_l2=0,
                  jitter=30,
                  lr=0.05,
                  reg_fn='DI',
                  targets=-1,
                  batch_size=50,
                  iterations=20000,
                  dataset='ImageNet',
                  model_name='ResNet50',
                  amp_mode='off',
                  seed=42,
                  output_dir='subnet experiments')
    DIargs = parse_DIargs(DIargs)
    return DIargs


def main():
    model = GetPretrainedSubnet(config='hidden_networks/configs/largescale/subnetonly/resnet50-usc-unsigned.yaml',
                                weights='hidden_networks/checkpoints/resnet50_usc_unsigned.pth')
    DIargs = GetDIargs()
    DIargs.model = model
    if DIargs.cuda:
        DIargs.model.cuda()

    DI = DeepInvert(**vars(DIargs))
    images = DI.deepInvert(**vars(DIargs))
    for i, image in enumerate(images):
        label = LABELS[DIargs.dataset][DIargs.targets[i].item()]
        image.save(os.path.join(DIargs.output_dir, f'{i}_{label}.png'))


if __name__ == '__main__':
    main()
