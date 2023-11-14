import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
from PIL import Image

class YODADataset(Dataset):
    def __init__(self, dir, training=True, transform=None):
        self.dir = dir
        self.training = training
        if self.training == False:
            self.dir = os.path.join(dir, 'train')
        else:
            self.dir = os.path.join(dir, 'test')
        
        self.label_dir = os.path.join(self.dir, 'labels.txt')
        
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
        image = Image.open(img_path).convert('RGB')
        if image is None:
            print(f"Warning: Could not read image {filename}")
            return None, None
        if self.transform:
            image = self.transform(image)
        label = int(self.class_label[filename])
        return image, label
