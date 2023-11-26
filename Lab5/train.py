import torch
from torchvision import datasets, transforms, models
import os
from tqdm import tqdm
import time
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pet_dataset import PetDataset

warnings.filterwarnings("ignore")

pet_dataset_mean = [0.4837, 0.4510, 0.3948]
pet_dataset_std = [0.2247, 0.2216, 0.2228]



def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    print('Using device:', device)

    train_dataset = PetDataset(args.data_dir, 'train')
    test_dataset = PetDataset(args.data_dir, 'test')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=4)

    # model being used to return x and y pos of pet nose
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
    elif args.arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    else:
        raise Exception('Architecture not supported!')
    
    model = model.to(device)
    criterion = torch.nn.MSELoss()

    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    scheduler = ReduceLROnPlateau(optim, 'min', patience=5, verbose=True)

    train_losses = []
    test_losses = []

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
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target.float())
                test_loss += loss.item()
        
        test_loss /= len(test_loader)

        scheduler.step(test_loss)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print('Train loss: %.4f, Test loss: %.4f' % (train_loss, test_loss))
        print('Time: %.2fs' % (time.time() - start_time))
        print()
     



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


    args = parser.parse_args()

    print('Parameters:')
    print('\tData Dir: %s' % args.data_dir)
    print('\tSave Dir: %s' % args.save_dir)
    print('\tArchitecture: %s' % args.arch)
    print('\tLearning Rate: %f' % args.learning_rate)
    print('\tBatch Size: %d' % args.batch_size)
    print('\tEpochs: %d' % args.epochs)
    print('\tUse Cuda: %s' % args.cuda)
    print()

    train(args)
