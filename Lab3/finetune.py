import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import time

import matplotlib.pyplot as plt

from vanilla import VanillaFrontend, VGG, ModFrontend

def plot_loss(loss_list, save_path):
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(save_path)

def main():

    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--classes', type=int, default=100)
    parser.add_argument('--encoder', type=str, default='encoder.pth')
    parser.add_argument('--s', type=str, default='encoder_finetuned.pth')
    parser.add_argument('--frontend', type=str, default='frontend.pth')
    parser.add_argument('--save_plot', type=str, default='loss2.png')
    parser.add_argument('--encoder_lr', type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device being used: {device}")

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(15),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    transforms.ToTensor(),  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    encoder = VGG.encoder
    encoder.load_state_dict(torch.load(args.encoder, map_location=device))
    for param in encoder.parameters():
        param.requires_grad = True
    encoder = encoder.to(device)
    
    #model = VanillaFrontend(encoder, num_classes=args.classes).to(device)
    model = ModFrontend(encoder, num_classes=args.classes).to(device)
    criterion = nn.CrossEntropyLoss()
    encoder_params = list(encoder.parameters())
    rest_of_model_params = list(model.parameters())[len(encoder_params):]
    optimizer = optim.SGD([
        {'params': encoder_params, 'lr': args.encoder_lr},
        {'params': rest_of_model_params, 'lr': args.lr}
    ], momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


    losses = []
    total_time = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        running_loss = 0.0
        num_batches = 0

        for i, data in enumerate(trainloader):

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
        scheduler.step(avg_loss)

    print ('Finished Training')
    print (f'Total time taken: {time.time() - total_time:.3f} seconds')
    torch.save(model.state_dict(), args.frontend)
    torch.save(encoder.state_dict(), args.s)
    plot_loss(losses, args.save_plot)

if __name__ == "__main__":
    main()