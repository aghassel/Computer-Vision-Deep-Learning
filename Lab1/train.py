import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import datetime
import argparse
from model import autoencoderMLP4Layer

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):
    """
    Trains model from model.py on MNIST dataset. Returns training losses.

    n_epochs (int): Number of Epochs
    optimizer (torch.optim): Optimizer, Adam is recommended
    model (torch.nn): PyTorch model
    loss_fn: record loss during training 
    train_loader (torch.utils.data.DataLoader): PyTorch Dataloader
    scheduler (): PyTorch scheduler. StepLR is recommended.
    device (torch.device): Processing Unit. CUDA GPU is recommended.
    """
    print('training...')
    model.to(device=device)
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

    return losses_train

def loss_plot(losses_train, save_path):
    """
    Plots losses from training vs epochs. Returns saved plot.

    losses_train (list): training losses
    save_path: Local path for plot
    """
    plt.plot(losses_train)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Autoencoder")
    parser.add_argument('-z', '--bottleneck', type=int, default=8, help="Bottleneck size")
    parser.add_argument('-e', '--epochs', type=int, default=50, help="Epochs")
    parser.add_argument('-b', '--batch_size', type=int, default=2048, help="Batch size")
    parser.add_argument('-s', '--save_model', type=str, default="MLP.8.pth", help="Save model")
    parser.add_argument('-p', '--loss_plot', type=str, default="loss.MLP.8.png", help="Save loss plot")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = autoencoderMLP4Layer(N_input=784, N_bottleneck=args.bottleneck, N_output=784)
    model.to(device=device)
    loss_fn = torch.nn.MSELoss(reduction="mean")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    losses_train = train(n_epochs=args.epochs, optimizer=optimizer, model=model, loss_fn=loss_fn,
          train_loader=train_loader, scheduler=scheduler, device=device)

    torch.save(model.state_dict(), args.save_model)
    loss_plot(losses_train, args.loss_plot)