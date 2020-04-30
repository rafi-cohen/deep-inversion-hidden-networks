from os import path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

from params import MEANS, STDS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_set_evaluation(test_dataset_loader, model):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    test_loss = 0
    dataset_size = len(test_dataset_loader.dataset)
    correct_predictions = 0
    for batch, targets in tqdm(test_dataset_loader):
        batch, targets = batch.to(device), targets.to(device)
        output = model(batch)
        loss = loss_fn(output, targets)
        test_loss += loss.item()
        probabilities = nn.functional.softmax(output, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        correct_predictions += sum(predictions == targets).item()
    test_accuracy = correct_predictions / dataset_size
    return test_loss, test_accuracy


def train_student_model(train_set_dir='dataset', test_set_dir='test_dataset', epochs=250, batch_size=32, lr=0.1):
    mean, std = MEANS['ImageNet'], STDS['ImageNet']
    train_preprocess = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std)])
    test_preprocess = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std)])
    train_dataset = datasets.ImageFolder(train_set_dir, transform=train_preprocess)
    test_dataset = datasets.ImageFolder(test_set_dir, transform=test_preprocess)
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # a randomly initialized ResNet50 R-CNN
    model = models.resnet50()
    targets_count = len(train_dataset.classes)
    cnn_features = model.fc.in_features
    # replace the fc part of the model so it would fit the targets count of our synthesized dataset
    model.fc = nn.Sequential(nn.Linear(cnn_features, 512, bias=True),
                             nn.ReLU(inplace=True),
                             nn.Linear(512, targets_count, bias=True))
    model = model.to(device)
    optimizer = Adam(params=model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=7, threshold=1e-2, verbose=True, min_lr=1e-6)
    loss_fn = nn.CrossEntropyLoss()

    train_loss_list = []
    test_loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    train_dataset_size = len(train_dataset)
    with tqdm(total=epochs, desc='Epoch:') as epoch_pbar:
        for epoch_idx in range(epochs):
            model.train()
            train_loss = 0
            correct_train_predictions = 0
            for batch, targets in tqdm(train_dataset_loader):
                batch, targets = batch.to(device), targets.to(device)
                output = model(batch)
                probabilities = nn.functional.softmax(output, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                correct_train_predictions += sum(predictions == targets).item()
                loss = loss_fn(output, targets)
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step(train_loss)
            train_loss_list.append(train_loss)
            train_accuracy = correct_train_predictions / train_dataset_size
            train_accuracy_list.append(train_accuracy)
            test_loss, test_accuracy = test_set_evaluation(test_dataset_loader, model)
            test_loss_list.append(test_loss)
            test_accuracy_list.append(test_accuracy)
            epoch_pbar.set_description_str(f'Epoch: {epoch_idx+1} Train Loss: {train_loss}'
                                           f' Train Accuracy: {train_accuracy}'
                                           f'Test Loss: {test_loss} Test Accuracy: {test_accuracy}')
            epoch_pbar.update()

    # save the trained model parameters:
    torch.save(model.state_dict(), path.join(train_set_dir, 'student_params'))
    # save test set accuracy & loss metrics per epoch
    with open(path.join(test_set_dir, 'test_loss_accuracy_results.txt'), 'w') as f:
        print('test set accuracy by epoch:', file=f)
        print(test_accuracy_list, file=f)
        print('test set loss by epoch:', file=f)
        print(test_loss_list, file=f)
        print('train set accuracy be epoch:', file=f)
        print(train_accuracy_list, file=f)
        print('train set loss by epoch', file=f)
        print(train_loss_list, file=f)


def main():
    train_student_model('dataset')


if __name__ == '__main__':
    main()
