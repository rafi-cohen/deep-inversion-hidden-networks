# Project structure
```text
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
```

# How to run
## To generate images using DeepInversion
Run the file `src/deep_inversion/main.py` with the wanted parameters.
The default parameters will generate 128 images of random labels in 20000 iterations.
To change that, for example, to generate 50 images of bears (label 294) in 2000 iterations, use:
```shell script
cd src/deep_inversion
python main.py --batch-size 50 --targets 294 --iterations 2000
```
Use `python main.py --help` to see the various options in our implementation

* **Note**: To run DeepInversion on CIFAR10, you must first download the weights corresponding to models trained on this dataset. To do so, run the file `src/deep_inversion/cifar10_models/cifar10_download.py`


## To run the grid search
Running the grid search used for the hyper-parameter tuning is as simple as:
```shell script
cd src/deep_inversion
python grid_search.py
```
This script will run all possible configurations according to a pre-defined grid,
will save the results generated using each configuration, as well as the inception score
corresponding to that configuration. Lastly, it will print the configuration which
resulted in the highest score.
To use a different grid, simply modify the `GRID` variable within the script.

## To synthesize a dataset of specific labels
The script for doing that is located at `src/deep_inversion/dataset_generation.py`. Again, to run use:
```shell script
cd src/deep_inversion
python dataset_generation.py
```
This script generates a dataset of 5000 images consisting of 10 different classes,
all of them are synthesized using DeepInversion.
To control the size of the generated dataset and the classes, you can modify
the variables `DATASET_SIZE` and `PARAMS['targets']` respectively within the script.

## To evaluate the generated dataset
This part requires the output of the dataset generation experiment.
Since generating the dataset takes a very long time, you can download
our results [here](https://drive.google.com/open?id=1-6vmNG2DAukQVvYETdRgs0tXIRNNKBm3)
Place this folder in `src/deep_inversion`. i.e., the structure should look like
```text
src/
└── deep_inversion/
    └── dataset/
        ├── label1/
        ├── label2/ 
        .
        .
        .
        └── labeln/
```
Then, run as usual:
```shell script
cd src/deep_inversion
python dataset_evaluation.py
```
This script will output the top-1/top-5 accuracy of a pretrained ResNet-152 on the dataset

# To run knowledge distillation
This part has the same requirements as the previous part (dataset evaluation),
but also needs a test dataset. The one we used is a subset of the original ImageNet
test dataset, and is available [here](https://drive.google.com/open?id=1ZPhalD29WeVkUHhgMfE2rXwmZFjdJNGY).
It should be located alongside the `dataset` folder.
To run:
```shell script
cd src/deep_inversion
python knowledge_distillation.py
```
This will train a ResNet-50 from scratch, and will output for each epoch:
test set accuracy, test set loss, train set accuracy, train set loss.
The results will be located at `test_loss_accuracy_results.txt`
A weights file named `student_params` will also be generated. To load it, use:
```python
import torch
from torchvision import models
model = models.resnet50()
model.load_state_dict(torch.load('student_params'))
```
# To invert a Hidden Subnetwork
This requires the the weights file of the pretrained subnetwork, supplied by
the authors of the paper. It can be downloaded at:
https://prior-pretrained-models.s3-us-west-2.amazonaws.com/hidden-networks/resnet50_usc_unsigned.pth
The file should be placed at `src/hidden_networks/checkpoints/resnet50_usc_unsigned.pth`
Then, run the experiment by:
```shell script
python subnet_inversion.py
```
This will generate 50 images by applying DeepInversion on the hidden subnetwork.