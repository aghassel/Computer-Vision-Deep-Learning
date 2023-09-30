#Modified version of train.py from https://github.com/naoto0804/pytorch-AdaIN

import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms
import AdaIN_net as net

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
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

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

    parser.add_argument('-gamma', type=float, help='lamda')
    parser.add_argument('-e', type=int, help='Number of epochs')
    parser.add_argument('-b', type=int, help='Batch size')
    parser.add_argument('-l', type=str, help='load encoder')
    parser.add_argument('-s', type=str, help='save decoder')
    parser.add_argument('-p', type=str, help='Path to save plot')
    parser.add_argument('-cuda', type=str, help='cuda', default='Y')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = net.encoder_decoder.encoder
    encoder.load_state_dict(torch.load(args.l, map_location=device))
    network = net.AdaIN_net(encoder)
    network.to(device)
    network.train()

    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)

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

    for epoch in range(args.e):
        adjust_learning_rate(optimizer, iteration_count=epoch)
        print(f"Epoch {epoch + 1}/{args.e}")
    
        for batch_idx, (content_images, style_images) in enumerate(zip(content_loader, style_loader)):
            content_images = content_images.to(device)
            style_images = style_images.to(device)
            
            optimizer.zero_grad()
            
            loss_c, loss_s = network(content_images, style_images)
            loss_c *= args.content_weight
            loss_s *= args.style_weight
            # Eq. (11)
            loss = loss_c + args.gamma*loss_s
            
            loss.backward()
            optimizer.step()
            
            content_losses.append(loss_c.item())
            style_losses.append(loss_s.item())
            total_losses.append(loss.item())
            
        avg_content_loss = sum(content_losses[-len(content_loader):]) / len(content_loader)
        avg_style_loss = sum(style_losses[-len(style_loader):]) / len(style_loader)
        avg_total_loss = sum(total_losses[-len(content_loader):]) / len(content_loader)

        avg_content_losses.append(avg_content_loss)
        avg_style_losses.append(avg_style_loss)
        avg_total_losses.append(avg_total_loss)

        print(f"Content Loss: {avg_content_loss:.2f}, Style Loss: {avg_style_loss:.2f}, Total Loss: {avg_total_loss:.2f}")

        torch.save(net.encoder_decoder.decoder.state_dict(), args.s) 
    
    loss_plot(avg_content_losses,avg_style_losses,avg_total_losses, args.p)           