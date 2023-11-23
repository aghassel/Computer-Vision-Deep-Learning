import torch
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class PetDataset(Dataset):
    def __init__ (self, dir, training=True, transform=None):
        self.dir = dir
        self.transform = transform
        self.training = training
  