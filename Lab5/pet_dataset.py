import torch
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from util import calculate_mean_std

resize_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

normalize_transform = None

def create_transforms(mean, std):
    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    normalize_transform = transforms.Normalize(mean, std)
    return transforms.Compose([resize_transform, normalize_transform])
    
class PetDataset(Dataset):
    def __init__ (self, dir, training=True, transform=None):
        self.dir = os.path.join(dir, 'images')
        self.transform = transform
        self.training = training
        
        if self.training:
            file_path = os.path.join(dir, 'train_noses.txt')
        else:
            file_path = os.path.join(dir, 'test_noses.txt')
        
        with open(file_path, 'r') as file:
            self.data = []
            for line in file:
                parts = line.strip().split(',', 1)
                self.data.append(parts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data[index]
        image_name, raw_label = line[0], line[1]

        image_path = os.path.join(self.dir, image_name)
        image = Image.open(image_path).convert('RGB')

        original_size = image.size
        label = [int(coord) for coord in raw_label.strip('"()').split(', ')]

        if self.transform is not None:
            image = self.transform(image)

        new_size = (224, 224)  
        scale_x, scale_y = new_size[0] / original_size[0], new_size[1] / original_size[1]
        label = [label[0] * scale_x, label[1] * scale_y]

        return image, torch.tensor(label, dtype=torch.float32)


if __name__ == "__main__":

    train_dataset = PetDataset(dir='data', training=True, transform=resize_transform)
    test_dataset = PetDataset(dir='data', training=False, transform=resize_transform)

    train_mean, train_std = calculate_mean_std(train_dataset)
    test_mean, test_std = calculate_mean_std(test_dataset)

    print('Calculated mean and std')
    print(f'Train mean: {train_mean}, Train std: {train_std}')
    print(f'Test mean: {test_mean}, Test std: {test_std}')

    normalize_transform = transforms.Normalize(train_mean, train_std)

    train_transform = transforms.Compose([resize_transform, normalize_transform])
    test_transform = transforms.Compose([resize_transform, normalize_transform])

    train_dataset = PetDataset(dir='data', training=True, transform=train_transform)
    test_dataset = PetDataset(dir='data', training=False, transform=test_transform)

    #Train mean: tensor([0.4789, 0.4476, 0.3948]), Train std: tensor([0.2259, 0.2229, 0.2255])
    #Test mean: tensor([0.4885, 0.4544, 0.3947]), Test std: tensor([0.2235, 0.2204, 0.2201])