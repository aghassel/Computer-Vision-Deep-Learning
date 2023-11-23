import torch
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from util import calculate_mean_std

#calculated from pet_dataset.py
pet_dataset_mean = [0.4837, 0.4510, 0.3948]
pet_dataset_std = [0.2247, 0.2216, 0.2228]

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
            lines = file.readlines()
        
        self.image_names = []
        self.labels = []
        for line in lines:
            image_name, label = line.strip().split(',', 1)
            label = label.strip('"()').split(', ')
            label = [int(coord) for coord in label]
            self.image_names.append(image_name)
            self.labels.append(label)
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.dir, self.image_names[index])
        image = Image.open(image_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        label = self.labels[index]
        
        return image, label

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = PetDataset(dir='data', training=True, transform=transform)
    test_dataset = PetDataset(dir='data', training=False, transform=transform)

    train_mean, train_std = calculate_mean_std(train_dataset)
    test_mean, test_std = calculate_mean_std(test_dataset)

    print('train mean: ', train_mean)
    print('train std: ', train_std)
    print('test mean: ', test_mean)
    print('test std: ', test_std)

    combined_mean = (train_mean + test_mean) / 2
    combined_std = (train_std + test_std) / 2

    print('combined mean: ', combined_mean)
    print('combined std: ', combined_std)
    



    