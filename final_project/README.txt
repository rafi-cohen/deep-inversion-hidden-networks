src/
├── deep_inversion/ - Package containing our implementation of DeepInversion
│   ├── deep_invert.py - Our implementation for DeepInversion as it is described in "Dreaming to Distill" (Yin et al., 2019)
│   ├── losses.py - Implementation of various loss functions for DeepInversion
│   ├── regularizations.py - Implementation of various regularization functions for DeepInversion
│   ├── transforms.py - Helper transforms used in DeepInversion
│   ├── params.py - Various hardcoded parameters used throughout our implementation
│   ├── parsing.py - Module for parsing arguments for DeepInversion
│   ├── main.py - Script for running DeepInversion to synthesize images
│   ├── cifar10_models/ - PyTorch implementation of various pretrained models trained on CIFAR10
│   │                     Taken from https://github.com/huyvnphan/PyTorch_CIFAR10
│   └── inception_score/ - PyTorch implementation of the Inception Score (Salimans et al., 2016)
│                          Taken from https://github.com/sbarratt/inception-score-pytorch
├── hidden_networks/ - PyTorch implementation of the edge-popup algorithm (Ramanujan et al., 2019)
│                      taken from https://github.com/allenai/hidden-networks
└── Experiments/ - Experiment scripts
    ├── grid_search.py - Hyper-parameter tuning for DeepInversion
    ├── dataset_generation.py - Invert 5000 images (from 10 classes) from a ResNet-50 pretrained on ImageNet
    ├── dataset_evaluation.py - Calculate top-1/top-5 accuracy on the above generated dataset using a pretrained ResNet-152
    ├── knowledge_distillation.py - Train a student ResNet-50 model from scratch using the above generated dataset
    └── subnet_inversion.py - Invert a pretrained well performing subnetwork of a randomly initialized ResNet-50


TODO: Steps to reproduce the results in the report:
