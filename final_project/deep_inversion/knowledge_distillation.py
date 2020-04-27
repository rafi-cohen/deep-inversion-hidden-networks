import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from os import path
from tqdm import tqdm
from params import MEANS, STDS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_student_model(dataset_path='dataset', epochs=40, batch_size=1, lr=1e-3):
    mean, std = MEANS['ImageNet'], STDS['ImageNet']
    transform_preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dataset = datasets.ImageFolder(dataset_path, transform=transform_preprocess)
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # a randomly initialized ResNet50 R-CNN
    model = models.resnet50().to(device)
    model.train()
    optimizer = Adam(params=model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    with tqdm(total=epochs, desc='Epoch:') as epoch_pbar:
        for epoch_idx in range(epochs):
            total_epoch_loss = 0
            for batch, targets in tqdm(dataset_loader):
                batch, targets = batch.to(device), targets.to(device)
                output = model(batch)
                loss = loss_fn(output, targets)
                total_epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_pbar.set_description_str(f'Epoch: {epoch_idx} Loss: {total_epoch_loss}')
            epoch_pbar.update()

    # save the trained model parameters:
    torch.save(model.state_dict(), path.join(dataset_path, 'student_params'))


def main():
    train_student_model('dataset')


if __name__ == '__main__':
    main()
