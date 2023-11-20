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
import matplotlib.pyplot as plt
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau
warnings.filterwarnings("ignore")



transform = transforms.Compose([
    transforms.Resize((150, 150)),  
    transforms.ToTensor(),
    transforms.Normalize([0.3656, 0.3844, 0.3725], [0.4194, 0.4075, 0.4239])
])


train_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomRotation(10),  
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(150, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    transforms.ToTensor(),
    transforms.Normalize([0.3656, 0.3844, 0.3725], [0.4194, 0.4075, 0.4239])
])



def train(args):
    mem_pin = False
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        mem_pin = True
    else:
        device = torch.device('cpu')
    print('Found device ', device)
    
    plot_dir = os.path.join(args.loss_plot_dir, args.model_name + '2')
    

    if args.model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, 1)
        print('Using resnet18')
    elif args.model_name == 'mod':
        encoder = VGG.encoder
        encoder.load_state_dict(torch.load('encoder.pth', map_location=device))
        model = ModFrontend(encoder)
        print('Using mod')
    else:
        print('Model not found, using resnet18')
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, 1)
        
    
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    weight = torch.tensor([args.pos_weight]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

    train_dataset = YODADataset(args.data_dir, training=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.threads, pin_memory=mem_pin)

    test_dataset = YODADataset(args.data_dir, training=False, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads, pin_memory=mem_pin)

    train_losses = []
    test_losses = []
    best_test_acc = 100000

    scheduler = StepLR(optimizer, step_size=5, gamma=0.95)
    reduce_lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

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
        reduce_lr_scheduler.step(test_loss)  
    
        print('Time: ', end-start, 's')

        if test_loss < best_test_acc:
            best_test_acc = test_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.model_name+'2.pth'))
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
    parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
    parser.add_argument('--pos_weight', type=float, default=3.0348, help='positive weight for BCEWithLogitsLoss (default: 3.0348, based off dataset imbalance)')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay (default: 0.0001)')
    args = parser.parse_args()

    plot_dir = os.path.join(args.loss_plot_dir, args.model_name)

    print('Parameters:')
    print('\tmodel_name:\t ', args.model_name)
    print('\tbatch_size:\t ', args.batch_size)
    print('\tepochs:\t\t ', args.epochs)
    print('\tlr:\t\t ', args.lr)
    print('\tpos_weight:\t ', args.pos_weight)
    print('\tmomentum:\t ', args.momentum)
    print('\tweight_decay:\t ', args.weight_decay)
    print('\tdata_dir:\t ', args.data_dir)
    print('\tsave_dir:\t ', args.save_dir)
    print('\tcuda:\t\t ', args.cuda)
    print('\tthreads:\t ', args.threads)
    print('\tloss_plot:\t ', plot_dir)
    print('\n')

    train(args)
