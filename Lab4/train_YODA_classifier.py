import argparse
import torch
from torchvision import datasets, transforms, models
import os
from torch.utils.data import Dataset
from YODADataset import YODADataset
from model import ModFrontend, VGG
from utils import plot_loss 
from tqdm import tqdm
import time
from torch.optim.lr_scheduler import StepLR

import warnings
warnings.filterwarnings("ignore")


def train(args):
    num_classes = 1
    if args.cuda and torch.cuda.is_available():

        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Found device ', device)
    
    plot_dir = os.path.join(args.loss_plot_dir, args.model_name)
    
    transform = transforms.Compose([
        transforms.Resize((150, 150)),  
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_train = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),  
        transforms.RandomRotation(15),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
        transforms.ToTensor(),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  
    ])

    if args.model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, num_classes)
    elif args.model_name == 'mod':
        encoder = VGG.encoder
        encoder.load_state_dict(torch.load('encoder.pth', map_location=device))
        model = ModFrontend(encoder, num_classes=num_classes)
    else:
        print('Model not found, using resnet18')
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, num_classes)
        
    
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_dataset = YODADataset(args.data_dir, training=True, transform=transform_train)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=3)

    test_dataset = YODADataset(args.data_dir, training=False, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=3)

    train_losses = []
    test_losses = []
    best_test_acc = 100000

    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    for epoch in range(args.epochs):
        print('Epoch: ', epoch)
        start = time.time()
        train_loss = 0.0
        test_loss = 0.0
        model.train()
        for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training: ', unit='batch', leave=False):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            #print(output.shape)
            #print(target.shape)
            loss = criterion(output, target.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)

        train_loss = train_loss/len(train_loader.dataset)
        train_losses.append(train_loss)
        print('Train Loss: ', train_loss)

        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing: ', unit='batch', leave=False):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target.unsqueeze(1).float())
                test_loss += loss.item()*data.size(0)

        end = time.time()

        test_loss = test_loss/len(test_loader.dataset)
        test_losses.append(test_loss)
        print('Test Loss: ', test_loss)
        scheduler.step()          
 
        print('Time: ', end-start, 's')

        if test_loss < best_test_acc:
            best_test_acc = test_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.model_name+'.pth'))
            print('Saved model')
        print(' ')
        plot_loss(train_losses, test_losses, plot_dir)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YODA classifier')
    parser.add_argument('--batch_size', type=int, default=75, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--model_name', type=str, default='resnet18', help='model name (default: resnet18)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--data_dir', type=str, default='data/Kitti8_ROIs/', help='path to dataset')
    parser.add_argument('--save_dir', type=str, default='models/', help='path to save trained model')
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
    parser.add_argument('--loss_plot_dir', type=str, default='resnet', help='path to save loss plots')
    args = parser.parse_args()

    plot_dir = os.path.join(args.loss_plot_dir, args.model_name)

    print('Parameters:')
    print('\tbatch_size: ', args.batch_size)
    print('\tepochs: ', args.epochs)
    print('\tlr: ', args.lr)
    print('\tmodel_name: ', args.model_name)
    print('\tmomentum: ', args.momentum)
    print('\tdata_dir: ', args.data_dir)
    print('\tsave_dir: ', args.save_dir)
    print('\tcuda: ', args.cuda)
    print('\tloss_plot: ', plot_dir)
    print('\n')

    train(args)
