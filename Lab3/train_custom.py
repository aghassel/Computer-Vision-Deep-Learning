import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR , ReduceLROnPlateau
import time


import matplotlib.pyplot as plt

#from arch.VGGResNet import VGG, VGG_w_skip
from arch.denseVGGResNet import VGG, DenseResNet152
#from model import VGG, ModFrontend

def plot_loss(loss_list, save_path):
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(save_path)

def main():

    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--classes', type=int, default=100)
    parser.add_argument('--encoder', type=str)
    parser.add_argument('--frontend', type=str)
    parser.add_argument('--save_plot', type=str)
    parser.add_argument('--resume_training', type=bool, default=False)

    args = parser.parse_args()

    print('Parameters')
    print(f'\tbatch size: {args.batch_size}')
    print(f'\tepochs: {args.epochs}')
    print(f'\tlr: {args.lr}')
    print(f'\tclasses: {args.classes}')
    print(f'\tencoder: {args.encoder}')
    print(f'\tfrontend: {args.frontend}')
    print(f'\tsave_plot: {args.save_plot}')
    print(f'\tresume_training: {args.resume_training}')
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device being used: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(15),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    transforms.ToTensor(),  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  
    ])

    #trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, \transform=transform)
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=9)
    valset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=9)

    encoder = VGG.encoder
    encoder.load_state_dict(torch.load(args.encoder, map_location=device))  
    encoder = encoder.to(device)
    
  
    model = DenseResNet152(encoder, args.classes).to(device)
    if args.resume_training:
        model.load_state_dict(torch.load(args.frontend, map_location=device))
        
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    for optimizer_ in optimizer.param_groups:
        optimizer_['lr'] = args.lr

    
    scheduler = StepLR(optimizer, step_size=5, gamma=0.95)

    losses = []
    patience = 0
    best_loss = float('inf')
    total_time = 0
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}')
        epoch_start = time.time()
        running_loss = 0.0
        model.train()
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()
        
        epoch_end = time.time()
        
        avg_loss = running_loss / len(trainloader)
        losses.append(avg_loss)
        print(f'Training loss: {avg_loss:.3f}, Time taken: {epoch_end - epoch_start:.3f} seconds')

        valtime = time.time()
        valloss = 0
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(valloader):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valloss += loss.item()
        valloss /= len(valloader)

        valtime = time.time() - valtime 

        print(f'Validation loss: {valloss:.3f}, Time taken: {valtime:.3f} seconds')
        if valloss < best_loss:
            print ('Saving model')
            best_loss = valloss
            torch.save(model.state_dict(), args.frontend) 
        print()

    print ('Finished Training')
    print (f'Total time taken: {time.time() - epoch_start:.3f} seconds')
    plot_loss(losses, args.save_plot)

if __name__ == "__main__":
    main()