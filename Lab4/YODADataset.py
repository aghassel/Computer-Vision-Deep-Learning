import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
import cv2

class YODADataset(Dataset):
    def __init__(self, dir, training=True, transform=None):
        self.dir = dir
        self.training = training
        if self.training == False:
            self.label_dir = os.path.join(dir, 'train')
        else:
            self.label_dir = os.path.join(dir, 'test')
        
        self.label_dir = os.path.join(dir, 'labels.txt')
        
        self.transform = transform
        self.img_files = []
        self.class_label = {}

        with open(self.label_dir, 'r') as f:
            for line in f:
                filename, label, _ = line.strip().split(' ')
                self.img_files.append(filename)
                self.class_label[filename] = label 

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        filename = self.img_files[idx]
        img_path = os.path.join(self.dir, filename)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if self.transform:
            image = self.transform(image)
        label = int(self.class_label[filename])
        return image, label

    def __iter__(self):
        self.num = 0
        return self

    def __next__(self):
        if (self.num >= self.max):
            raise StopIteration
        else:
            self.num += 1
            return self.__getitem__(self.num-1)
    