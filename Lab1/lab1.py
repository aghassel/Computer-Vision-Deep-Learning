import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import autoencoderMLP4Layer

# Step 4
def reconstructed_plot(loader):
    imgs, _ = next(iter(loader))

    with torch.no_grad():
        outputs = model(imgs.view(-1, 784).type(torch.float32))

        for i in range(imgs.size(0)):
            img = imgs[i].view(28, 28).numpy()
            output = outputs[i].view(28, 28).numpy()

            f = plt.figure()
            f.add_subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
            f.add_subplot(1, 2, 2)
            plt.imshow(output, cmap='gray')
            plt.show()

# Step 5
def noisy_reconstructed_plot(loader, noise_factor=0.25):
    imgs, _ = next(iter(loader))

    with torch.no_grad():
        
        noisy_imgs = imgs + noise_factor * torch.randn(*imgs.shape)
        noisy_imgs = torch.clamp(noisy_imgs, 0., 1.)
        outputs = model(noisy_imgs.view(-1, 784).type(torch.float32))

        fig, axes = plt.subplots(3, 3, figsize=(9, 9))

        for i in range(imgs.size(0)):
            img = imgs[i].view(28, 28).numpy()
            noisy_img = noisy_imgs[i].view(28, 28).numpy()
            output = outputs[i].view(28, 28).numpy()

            images = [img, noisy_img, output]
            
            for j, image in enumerate(images):
                axes[i, j].imshow(image, cmap='gray')

        plt.tight_layout()
        plt.show()

# Step 6
def bottleneck_interpolations(img1, img2, n_steps=8):
    with torch.no_grad():
        img1_bottleneck = model.encode(img1.view(1, 784).type(torch.float32))
        img2_bottleneck = model.encode(img2.view(1, 784).type(torch.float32))

        alphas = torch.linspace(0, 1, n_steps+2)

        interpolations, outputs = [], []

        for i, alpha in enumerate(alphas):
            interpolations.append((1-alpha) * img1_bottleneck + alpha * img2_bottleneck)
            outputs.append(model.decode(interpolations[i]).view(28, 28))
        
        fig, axes = plt.subplots(1, n_steps+2, figsize=(15, 2)) #added 2 since beginning and end are not interpolated
        for ax, img in zip(axes, outputs):
            ax.imshow(img, cmap='gray')

        plt.show()

if __name__ == "__main__":

    # Step 2
    # train_transform = transforms.Compose([transforms.ToTensor()])
    # train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)

    # idx = int(input("Enter an integer between 0 and 59999: "))
    # plt.imshow(train_set.data[idx], cmap='gray')
    # plt.title("Ground Truth: {}".format(train_set.targets[idx]))
    # plt.show()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_transform = transforms.Compose([transforms.ToTensor()])
    test_set = MNIST('./data/mnist', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=3, shuffle=False) # batch size of 3 is used here for visualization purposes

    model = autoencoderMLP4Layer(N_input=784, N_bottleneck=8, N_output=784)
    model.load_state_dict(torch.load('MLP.8.pth', map_location=torch.device(device)))
    model.eval()

    reconstructed_plot(test_loader)
    noisy_reconstructed_plot(test_loader)

    loader = DataLoader(test_set, batch_size=1, shuffle=True)
    img1, _ = next(iter(loader))
    img2, _ = next(iter(loader))

    img1 = img1.squeeze().view(784).type(torch.float32)
    img2 = img2.squeeze().view(784).type(torch.float32)

    bottleneck_interpolations(img1, img2, 8)
