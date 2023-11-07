# accuracy function from https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840

import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import time
import matplotlib.pyplot as plt

from model import VanillaFrontend, VGG, ModFrontend
from fcnresnet import DenseFCNResNet152

def plot_loss(loss_list, save_path):
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(save_path)
    plt.show()

def accuracy(output, target, topk=(1,)):

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--classes', type=int, default=100)
    parser.add_argument('--encoder', type=str, default='encoder.pth')
    parser.add_argument('--frontend', type=str, default=f'frontend.pth')
    parser.add_argument('--save_plot', type=str, default=f'loss.png')
    parser.add_argument('--model', type=str, default='resnet')
    args = parser.parse_args()

    print('Parameters')
    print(f'\tbatch size: {args.batch_size}')
    print(f'\tepochs: {args.epochs}')
    print(f'\tlr: {args.lr}')
    print(f'\tclasses: {args.classes}')
    print(f'\tencoder: {args.encoder}')
    print(f'\tfrontend: {args.frontend}')
    print(f'\tsave_plot: {args.save_plot}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device being used: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),  
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(15),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    transforms.ToTensor(),  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    encoder = VGG.encoder
    encoder.load_state_dict(torch.load(args.encoder, map_location=device))  
    encoder = encoder.to(device)
    
    if args.model == 'vanilla':
        model = VanillaFrontend(encoder, num_classes=args.classes).to(device)
    elif args.model == 'mod':
        model = ModFrontend(encoder, num_classes=args.classes).to(device)
    elif args.model == 'resnet':
        model = DenseFCNResNet152(encoder, num_classes=args.classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.95)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    #scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    losses = []
    total_time = 0

    best_top1_accuracy = 0.0
    best_top5_accuracy = 0.0
    best_model_state = None

    for epoch in range(args.epochs):
        
        running_loss = 0.0
        num_batches = 0
        epoch_start = time.time()

        for i, data in enumerate(trainloader, 0):

            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            num_batches += 1

        epoch_end = time.time()
        
        avg_loss = running_loss / num_batches
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Training loss: {avg_loss:.3f}, Time taken: {epoch_end - epoch_start:.3f} seconds')
        total_time += epoch_end - epoch_start
        scheduler.step()

        model.eval()

        top1_accuracy = 0.0
        top5_accuracy = 0.0
        num_batches = 0  

        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                images, labels = data[0].to(device), data[1].to(device) 
                outputs = model(images)
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                top1_accuracy += acc1[0]
                top5_accuracy += acc5[0]
                num_batches += 1

        top1_avg_accuracy = top1_accuracy / num_batches
        top5_avg_accuracy = top5_accuracy / num_batches

        if top1_avg_accuracy > best_top1_accuracy:
            best_top1_accuracy = top1_avg_accuracy
            best_top5_accuracy = top5_avg_accuracy
            best_model_state = model.state_dict()
        print(f'Epoch {epoch+1}, Top 1 accuracy: {top1_avg_accuracy:.2f}%, Top 5 accuracy: {top5_avg_accuracy:.2f}%')
        model.train()

    print('Finished Training')
    print(f'Total time taken: {total_time:.3f} seconds')
    
    torch.save(best_model_state, args.frontend)
    print(f"Best model saved with Top-1 accuracy: {best_top1_accuracy:.2f}%, Top-5 accuracy: {best_top5_accuracy:.2f}%")
        
    plot_loss(losses, args.save_plot)