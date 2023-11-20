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
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
warnings.filterwarnings("ignore")



transform = transforms.Compose([
    transforms.Resize((150, 150)),  
    transforms.ToTensor(),
    transforms.Normalize([0.3656, 0.3844, 0.3725], [0.4194, 0.4075, 0.4239])
])


def evaluate(args):
    mem_pin = False
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        mem_pin = True
    else:
        device = torch.device('cpu')
    print('Found device ', device)
        

    if args.model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, 1)
        print('Using resnet18')
    elif args.model_name == 'mod' or args.model_name == 'mod2':
        encoder = VGG.encoder
        encoder.load_state_dict(torch.load('encoder.pth', map_location=device))
        model = ModFrontend(encoder)
        print('Using mod')
    else:
        print('Model not found, using resnet18')
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, 1)

    model_path = os.path.join('models', args.model_name + '.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
        
    
    model.to(device)

    test_dataset = YODADataset(args.data_dir, training=False, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads, pin_memory=mem_pin)

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = torch.sigmoid(output)
            pred = torch.round(pred)
            y_true.append(target.item())
            y_pred.append(pred.item())
    
    print('Accuracy: ', accuracy_score(y_true, y_pred))
    print('F1 score: ', f1_score(y_true, y_pred))
    print('Confusion matrix: ')
    print(confusion_matrix(y_true, y_pred))
    print('Classification report: ')
    print(classification_report(y_true, y_pred))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YODA classifier')
    parser.add_argument('--batch_size', type=int, default=75, help='input batch size for training (default: 32)')
    parser.add_argument('--model_name', type=str, default='mod2', help='model name (default: resnet18)')
    parser.add_argument('--data_dir', type=str, default='data/Kitti8_ROIs/', help='path to dataset')
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
    parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
    args = parser.parse_args()


    print('Parameters:')
    print('\tmodel_name:\t ', args.model_name)
    print('\tdata_dir:\t ', args.data_dir)
    print('\tbatch_size:\t ', args.batch_size)
    print('\tthreads:\t ', args.threads)
    print('\tcuda:\t\t ', args.cuda) 
    print('\n')

    evaluate(args)
