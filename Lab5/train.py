import torch
from torchvision import models, transforms
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pet_dataset import PetDataset, create_transforms
from util import plot_loss
import argparse

warnings.filterwarnings("ignore")

def train(args):

    # mean and std are generated from pet_dataset.py
    train_mean = [0.4789, 0.4476, 0.3948]
    train_std = [0.2259, 0.2229, 0.2255]
    test_mean = [0.4885, 0.4544, 0.3947]
    test_std = [0.2235, 0.2204, 0.2201]
    
    train_transform = create_transforms(train_mean, train_std)
    test_transform = create_transforms(test_mean, test_std)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print('Using device:', device)

    train_dataset = PetDataset(args.data_dir, training = True, transform = train_transform)
    test_dataset = PetDataset(args.data_dir, training = False, transform = test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
    elif args.arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    else:
        raise ValueError('Only VGG16 and ResNet18 are supported!')

    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optim, 'min', patience=5, verbose=True)

    train_losses = []
    test_losses = []
    best_train_loss = float('inf')

    for epoch in range(args.epochs):
        print('Epoch:', epoch)
        start_time = time.time()

        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optim.zero_grad()
            output = model(data)
            loss = criterion(output, target.float())
            loss.backward()
            optim.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in tqdm(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target.float())
                test_loss += loss.item()

        test_loss /= len(test_loader)

        scheduler.step(test_loss)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f'Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}')
        print(f'Time: {time.time() - start_time:.2f}s')
        print()

        if args.save_dir:
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_best.pth'))

    plot_loss(train_losses, test_losses, args.plot_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='dataset directory')
    parser.add_argument('--save_dir', type=str, default='checkpoint/vgg', help='checkpoint directory')
    parser.add_argument('--arch', type=str, default='vgg16', help='model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--cuda',  type=bool, default=True, help='use cuda if available')
    parser.add_argument('--plot', type=str, default='plot.png', help='plot path')
    args = parser.parse_args()

    print('Parameters:')
    for arg in vars(args):
        print(f'\t{arg.capitalize()}: {getattr(args, arg)}')
    print()

    train(args)