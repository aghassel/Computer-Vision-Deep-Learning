import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import datetime
import argparse
from model import autoencoderMLP4Layer
from torchvision import transforms;
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):
    print('training...')
    model.train()

    losses_train = []
    for epoch in range(1, n_epochs + 1):
        print('epoch ', epoch)
        loss_train = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.view(imgs.size(0), -1)
            imgs = imgs.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step(loss_train)
        losses_train += [loss_train / len(train_loader)]
        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / len(train_loader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an autoencoder for MNIST.")
    parser.add_argument('-e', '--epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('-z', '--bottleneck', type=int, default=8, help="Bottleneck size")
    parser.add_argument('-b', '--batch_size', type=int, default=392, help="Batch size")
    parser.add_argument('-s', '--save_path', type=str, default="MLP.8.pth", help="Save path for the model")
    parser.add_argument('-p', '--loss_graph_path', type=str, default="loss.MLP.8.png", help="Loss graph save location")
    args = parser.parse_args()

    model = autoencoderMLP4Layer(N_input=784, N_bottleneck=args.bottleneck, N_output=784)
    loss_fn = torch.nn.MSELoss(reduction="mean")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    train(n_epochs=args.epochs, optimizer=optimizer, model=model, loss_fn=loss_fn,
          train_loader=train_loader, scheduler=scheduler, device="cpu")

    torch.save(model.state_dict(), args.save_path)