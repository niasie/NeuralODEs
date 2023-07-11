import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def transformations():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    return transform_train, transform_test


def get_datasets(transform_train, transform_test):
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                             transform=transform_train)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                            transform=transform_test)

    return train_dataset, test_dataset


def get_dataloader(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_accuracy(logit, target, batch_size):
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects / batch_size
    return accuracy.item()