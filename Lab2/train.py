#Modified version of train.py from https://github.com/naoto0804/pytorch-AdaIN

import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
import os
from torchvision import transforms
import AdaIN_net as net
import time

import matplotlib.pyplot as plt

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True



def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        if 'wikiart' in root:
            self.paths = list(Path(self.root).glob('*/*.jpg'))
            print('Wikiart dataset length: ', len(self.paths))
        else:
            self.paths = list(Path(self.root).glob('*'))
            print('COCO dataset length: ', len(self.paths))

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def loss_plot(content_losses, style_losses,total_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(total_losses, label='Content + Style', color='blue')
    plt.plot(content_losses, label='Content', color='orange')
    plt.plot(style_losses, label='Style', color='green')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AdaIN Style Transfer")
    # Basic options

    parser.add_argument('-content_dir', type=str, required=True,
                        help='Directory path to a batch of content images')
    parser.add_argument('-style_dir', type=str, required=True,
                        help='Directory path to a batch of style images')
    # training options
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--style_weight', type=float, default=10.0)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--n_threads', type=int, default=2)

    parser.add_argument('-gamma', type=float, help='lamda used in Eq. 11')
    parser.add_argument('-e', type=int, help='Number of epochs')
    parser.add_argument('-b', type=int, help='Batch size')
    parser.add_argument('-l', type=str, help='load encoder')
    parser.add_argument('-s', type=str, help='save decoder')
    parser.add_argument('-p', type=str, help='Path to save plot')
    parser.add_argument('-cuda', type=str, help='cuda', default='Y')
    args = parser.parse_args()

    print("Args:", args)

    print('Cuda available: ', torch.cuda.is_available())

    device = torch.device("cuda")

    encoder = net.encoder_decoder.encoder
    encoder.load_state_dict(torch.load(args.l, map_location=device))
    network = net.AdaIN_net(encoder)
    network.to(device)
    network.train()

    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)

    num_images = 10000

    content_dataset = torch.utils.data.Subset(content_dataset, list(range(num_images)))
    style_dataset = torch.utils.data.Subset(style_dataset, list(range(num_images)))

    first_content_image = content_dataset[0].to(device)
    print("First content image shape: ", first_content_image.shape)

    first_style_image = style_dataset[0].to(device)
    print("First style image shape: ", first_style_image.shape)

   
    # content_iter = iter(data.DataLoader(
    #     content_dataset, batch_size=args.b,
    #     sampler=InfiniteSamplerWrapper(content_dataset),
    #     num_workers=args.n_threads))
    # style_iter = iter(data.DataLoader(
    #     style_dataset, batch_size=args.b,
    #     sampler=InfiniteSamplerWrapper(style_dataset),
    #     num_workers=args.n_threads))

    content_loader = data.DataLoader(content_dataset, batch_size=args.b, shuffle=True, num_workers=args.n_threads)
    style_loader = data.DataLoader(style_dataset, batch_size=args.b, shuffle=True, num_workers=args.n_threads)

    content_losses = []
    style_losses = []
    total_losses = []
  
    avg_content_losses = []
    avg_style_losses = []
    avg_total_losses = []

    optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

    print('Content Dataset length: ', len(content_dataset), ', Style Dataset length: ', len(style_dataset))
    print("Batch size: ", args.b)
    print("Content length: ", len(content_dataset), ' / ', args.b, ' = ', len(content_loader))
    print("Style length: ", len(style_dataset), ' / ', args.b, ' = ', len(style_loader))

    print("Device", device)

    for epoch in range(args.e):
        adjust_learning_rate(optimizer, iteration_count=epoch)
        print(f"Epoch {epoch + 1}/{args.e}")
        epoch_content_loss = 0
        epoch_style_loss = 0
        epoch_total_loss = 0
        epoch_start = time.time()
        count = 0

        for content_images, style_images in zip(content_loader, style_loader):
            content_images = content_images.to(device)
            style_images = style_images.to(device)

            optimizer.zero_grad()

            loss_c, loss_s = network(content_images, style_images)
            # loss_c *= args.content_weight # Default is 1
            # loss_s *= args.style_weight # Default is 10, to ensure the generated image reflects more of the style than content
            # Eq. (11)
            loss = loss_c + args.gamma * loss_s

            loss.backward()
            optimizer.step()

            content_losses.append(loss_c.item())
            style_losses.append(loss_s.item())
            total_losses.append(loss.item())

            epoch_content_loss += loss_c.item()
            epoch_style_loss += loss_s.item()
            epoch_total_loss += loss.item()

        epoch_end = time.time()
            
        avg_content_loss = sum(content_losses[-len(content_loader):]) / len(content_loader)
        avg_style_loss = sum(style_losses[-len(style_loader):]) / len(style_loader)
        avg_total_loss = sum(total_losses[-len(content_loader):]) / len(content_loader)

        avg_content_losses.append(avg_content_loss)
        avg_style_losses.append(avg_style_loss)
        avg_total_losses.append(avg_total_loss)

        print(f"Content Loss: {avg_content_loss:.2f}, Style Loss: {avg_style_loss:.2f}, Total Loss: {avg_total_loss:.2f}\n")
        print(f"Time taken for epoch: {epoch_end - epoch_start:.2f} seconds\n")

        torch.save(net.encoder_decoder.decoder.state_dict(), args.s) 
    
    loss_plot(avg_content_losses,avg_style_losses,avg_total_losses, args.p)           