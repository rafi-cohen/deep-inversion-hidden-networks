import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from os import path
from tqdm import tqdm
from params import MEANS, STDS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_set_evaluation(test_dataset_loader, model):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    test_loss = 0
    results = []
    for batch, targets in tqdm(test_dataset_loader):
        batch, targets = batch.to(device), targets.to(device)
        output = model(batch)
        loss = loss_fn(output, targets)
        test_loss += loss.item()
        probabilities = nn.functional.softmax(output, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        results.extend(predictions == targets)
    test_accuracy = sum(results) / len(test_dataset_loader)
    model.train()
    return test_loss, test_accuracy


def train_student_model(train_set_dir='dataset', test_set_dir='test_dataset', epochs=100, batch_size=64, lr=0.1):
    mean, std = MEANS['ImageNet'], STDS['ImageNet']
    transform_preprocess = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std)])
    train_dataset = datasets.ImageFolder(train_set_dir, transform=transform_preprocess)
    test_dataset = datasets.ImageFolder(test_set_dir, transform=transform_preprocess)
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    # a randomly initialized ResNet50 R-CNN
    model = models.resnet50().to(device)
    model.train()
    optimizer = Adam(params=model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=5, threshold=1e-2, verbose=True, min_lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    test_loss_list = []
    test_accuracy_list = []
    with tqdm(total=epochs, desc='Epoch:') as epoch_pbar:
        for epoch_idx in range(epochs):
            total_epoch_loss = 0
            for batch, targets in tqdm(train_dataset_loader):
                batch, targets = batch.to(device), targets.to(device)
                output = model(batch)
                loss = loss_fn(output, targets)
                total_epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step(total_epoch_loss)
            test_loss, test_accuracy = test_set_evaluation(test_dataset_loader, model)
            test_loss_list.append(test_loss)
            test_accuracy_list.append(test_accuracy)
            epoch_pbar.set_description_str(f'Epoch: {epoch_idx} Train Loss: {total_epoch_loss} '
                                           f'Test Loss {test_loss} Test Accuracy {test_accuracy}')
            epoch_pbar.update()

    # save the trained model parameters:
    torch.save(model.state_dict(), path.join(train_set_dir, 'student_params'))
    # save test set accuracy & loss metrics per epoch
    with open(path.join(test_set_dir, 'test_loss_accuracy_results.txt'), 'w') as f:
        print('test set accuracy by epoch:', file=f)
        print(test_accuracy_list, file=f)
        print('test set loss by epoch:', file=f)
        print(test_loss_list, file=f)


def main():
    train_student_model('dataset')


if __name__ == '__main__':
    main()
