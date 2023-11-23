import torch
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

mean = torch.tensor([0.3656, 0.3844, 0.3725])
std = torch.tensor([0.4194, 0.4075, 0.4239])


def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

class YODADataset(Dataset):
    def __init__(self, dir, training=True, transform=None):
        self.dir = dir
        self.training = training
        if self.training == True:
            self.dir = os.path.join(dir, 'train')
        else:
            self.dir = os.path.join(dir, 'test')
        
        self.label_dir = os.path.join(self.dir, 'labels.txt')
        
        
        self.transform = transform
        self.data = [] 

        with open(self.label_dir, 'r') as f:
            for line in f:
                filename, label, _ = line.strip().split(' ')
                full_path = os.path.join(self.dir, filename)
                self.data.append((full_path, int(label)))  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_path, label = self.data[idx]
        image = Image.open(full_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        #plt.imshow(image.permute(1, 2, 0))
        #plt.show()

        return image, label
    

if __name__ == '__main__':
    import torch
    from tqdm import tqdm

    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ])

    train_dataset = YODADataset(dir='data/Kitti8_ROIs', training=True, transform=transform)
    test_dataset = YODADataset(dir='data/Kitti8_ROIs', training=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_samples = 0

    class_counts = {0: 0, 1: 0}  

    for images, labels in tqdm(train_loader):
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean((0, 2)) * batch_size
        std += images.std((0, 2)) * batch_size
        num_samples += batch_size

        # Count class occurrences
        for label in labels:
            if label == 1:
                class_counts[1] += 1
            else:
                class_counts[0] += 1
    
    for images, labels in tqdm(test_loader):
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean((0, 2)) * batch_size
        std += images.std((0, 2)) * batch_size
        num_samples += batch_size

        # Count class occurrences
        for label in labels:
            if label == 1:
                class_counts[1] += 1
            else:
                class_counts[0] += 1

    mean /= num_samples
    std = torch.sqrt(std / num_samples - mean ** 2)

    print("Mean:", mean)
    print("Std:", std)


    print ('Number of images with cars:', class_counts[1])
    print ('Number of images without cars:', class_counts[0])
    # Calculate class ratios
    class_ratios = {0: class_counts[0] / num_samples, 1: class_counts[1] / num_samples}
    print("Class Ratios:", class_ratios)

    # Calculate pos_weight
    pos_weight = torch.tensor([class_ratios[0] / class_ratios[1]])
    print("Pos Weight:", pos_weight)
    