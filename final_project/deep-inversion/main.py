import torch
from os import path
from PIL import Image
from deep_inversion import DeepInvert
from torchvision import transforms

CUDA_ENABLED = torch.cuda.is_available()
if CUDA_ENABLED:
  torch.cuda.empty_cache()
# vgg11_bn
# resnet18
model_name = 'resnet18'
VGG11_BN_PARAMS = dict(model_name='vgg11_bn',
                       image_size=(32, 32),
                       mean=[0.4914, 0.4822, 0.4465],
                       std=[0.2023, 0.1994, 0.2010],
                       batch_size=30,
                       target_class=8,
                       lr=0.05,
                       a_tv=2.5e-5,
                       a_l2=3e-8,
                       a_f=1,
                       a_c=0.2)
RESNET34_PARAMS = dict(model_name='resnet34',
                       image_size=(32, 32),
                       mean=[0.4914, 0.4822, 0.4465],
                       std=[0.2023, 0.1994, 0.2010],
                       batch_size=30,
                       target_class=5,
                       lr=0.05,
                       a_tv=2.5e-5,
                       a_l2=3e-8,
                       a_f=1,
                       a_c=0.2)
RESNET18_PARAMS = dict(model_name='resnet18',
                       image_size=(224, 224),
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       batch_size=10,
                       target_class=950,
                       lr=0.05,
                       a_tv=1e-6,
                       a_l2=0,
                       a_f=1e-2,
                       a_c=0.2)

PARAMS_DICT = dict(vgg11_bn=VGG11_BN_PARAMS,
                   resnet34=RESNET34_PARAMS,
                   resnet18=RESNET18_PARAMS)

HYPERPARAMS = PARAMS_DICT[model_name]

OUT_DIR = 'generated'
FILENAME_FORMAT = path.join(OUT_DIR, '{}.jpg')


def main():
    # Load a local image:
    # batch = Image.open(path.join(OUT_DIR, '9.jpg'))
    # preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    # batch = preprocess(batch).unsqueeze(0)

    # Create random noise:
    batch = torch.randn(HYPERPARAMS['batch_size'], 3, *HYPERPARAMS['image_size'])

    target_criterion = HYPERPARAMS['target_class'] * torch.ones(HYPERPARAMS['batch_size'], dtype=torch.long)
    if CUDA_ENABLED:
        batch = batch.cuda()
        target_criterion = target_criterion.cuda()
    DI = DeepInvert(HYPERPARAMS['model_name'], HYPERPARAMS['mean'], HYPERPARAMS['std'], CUDA_ENABLED,
                    HYPERPARAMS['a_tv'], HYPERPARAMS['a_l2'], HYPERPARAMS['a_f'])
    images = DI.deepInvert(batch, iterations=2000,
                           target_criterion=target_criterion, lr=HYPERPARAMS['lr'])
    for i, image in enumerate(images):
        image.save(FILENAME_FORMAT.format(i))


if __name__ == '__main__':
    main()
