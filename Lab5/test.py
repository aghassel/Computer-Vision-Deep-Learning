import torch
from torchvision import models, transforms
from torchvision.utils import make_grid
import argparse
from pet_dataset import PetDataset, create_transforms
import matplotlib.pyplot as plt
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

def load_model(model_path, arch, device):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    else:
        raise ValueError('Architecture not supported!')
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def test(model, test_loader, device):
    model.eval()
    images = []
    predicted_coordinates = []
    ground_truth_coords = []
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            output = model(data)
            predicted_coordinates.extend(output.cpu().numpy())
            images.extend(data.cpu())
            ground_truth_coords.extend(targets.cpu().numpy())
    return images, predicted_coordinates, ground_truth_coords


def unnormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# def show_predictions(images, predicted_coords, num_images=10, mean=None, std=None):
#     if len(images) > num_images:
#         images = images[:num_images]
#         predicted_coords = predicted_coords[:num_images]

#     plt.figure(figsize=(15, 10))
#     plt.subplots_adjust(wspace=0.1, hspace=0.1)
#     for i in range(len(images)):
#         img = images[i]
#         if mean is not None and std is not None:
#             img = unnormalize(img, mean, std)
#         img = transforms.ToPILImage()(img)
#         plt.subplot(2, 5, i + 1)
#         plt.imshow(img)
#         x, y = predicted_coords[i]
#         plt.scatter(x, y, c='red', s=40, marker='x', edgecolors='white')
#         plt.annotate('Nose', (x,y), textcoords="offset points", xytext=(0,10), ha='center', color='black')
#         plt.title(f'Image {i+1}')
#         plt.axis('off')
#     plt.show()
#     plt.tight_layout()
#     plt.savefig('predictions.png')

def show_predictions(images, predicted_coords, ground_truth_coords, num_images=10, mean=None, std=None):
    if len(images) > num_images:
        images = images[:num_images]
        predicted_coords = predicted_coords[:num_images]
        ground_truth_coords = ground_truth_coords[:num_images]

    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    for i in range(len(images)):
        img = images[i]
        if mean is not None and std is not None:
            img = unnormalize(img, mean, std)
        img = transforms.ToPILImage()(img)
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        pred_x, pred_y = predicted_coords[i]
        true_x, true_y = ground_truth_coords[i]
        plt.scatter(pred_x, pred_y, c='red', s=40, marker='x', edgecolors='white', label='Predicted')
        plt.scatter(true_x, true_y, c='blue', s=40, marker='o', edgecolors='white', label='Ground Truth')
        plt.title(f'Image {i+1}')
        plt.axis('off')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='dataset directory')
    parser.add_argument('--model_path', type=str, default='checkpoint/resnet/model_best.pth', help='path to the trained model')
    parser.add_argument('--arch', type=str, default='resnet18', help='model architecture')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--cuda',  type=bool, default=True, help='use cuda if available')
    args = parser.parse_args()

    #Pre-calculated mean and std
    test_mean = [0.4885, 0.4544, 0.3947]
    test_std = [0.2235, 0.2204, 0.2201]

    test_transform = create_transforms(test_mean, test_std)
    test_dataset = PetDataset(args.data_dir, training=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print('Using device:', device)
    model = load_model(args.model_path, args.arch, device)
    
    images, predicted_nose_coords, ground_truth_coords = test(model, test_loader, device)
    show_predictions(images, predicted_nose_coords, ground_truth_coords, num_images=10, mean=test_mean, std=test_std)
