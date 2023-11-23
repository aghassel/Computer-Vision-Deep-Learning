import torch
from torchvision import datasets, transforms, models
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def calculate_mean_std(dataset):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_samples = len(dataset)

    for image, _ in tqdm(dataset, desc='Calculating mean and std', unit='images', leave=False):
        mean += torch.mean(image, dim=(1, 2))
        std += torch.std(image, dim=(1, 2))

    mean /= num_samples
    std /= num_samples

    return mean, std


def plot_loss(train_loss, val_loss, save_path):
    plt.figure()
    plt.plot(train_loss, label='train loss')
    plt.plot(val_loss, label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(save_path)
