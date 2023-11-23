import torch
from torchvision import datasets, transforms, models
import os
from torch.utils.data import Dataset
from tqdm import tqdm
import time
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau
warnings.filterwarnings("ignore")




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='dataset directory')
    parser.add_argument('--save_dir', type=str, default='checkpoint', help='checkpoint directory')
    parser.add_argument('--arch', type=str, default='vgg16', help='architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')
    parser.add_argument('--epochs', type=int, default=30, help='epochs')
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda or not')
