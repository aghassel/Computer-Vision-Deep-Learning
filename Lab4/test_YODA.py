from model import ModFrontend, VGG
import os
import torch
from torchvision import datasets, transforms, models
from Kitti.KittiDataset import KittiDataset
from Kitti.KittiAnchors import Anchors
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def test(args):
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

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

    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.3656, 0.3844, 0.3725], [0.4194, 0.4075, 0.4239])
    ])

    dataset = KittiDataset(args.data_dir, training=False)
    anchors = Anchors()

    model.eval()

    ious = []


    with torch.no_grad():
        for item in enumerate(dataset):
            idx = item[0]
            image = item[1][0]
            label = item[1][1]

            idx = dataset.class_label['Car']
            car_ROIs = dataset.strip_ROIs(class_ID=idx, label_list=label)

            anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)
            anchor_ROIs, boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)

            image_copy = image.copy()

            ROI_IoUs = []
            for idx in range(len(anchor_ROIs)):
                ROI = anchor_ROIs[idx]
                ROI = Image.fromarray(ROI)
                ROI = transform(ROI)
                ROI = ROI.unsqueeze(0)
                ROI = ROI.to(device)

                output = model(ROI)
                output = torch.sigmoid(output)
                output = output.item()

                if output > IoU_threshold:
                    ROI_IoUs += [anchors.calc_IoU(boxes[idx], car_ROIs)]
                else:
                    ROI_IoUs += [0]
            
            ious += [max(ROI_IoUs)]

            print('Image: ', idx, ' IoU: ', max(ROI_IoUs))
        
        print('Average IoU: ', sum(ious)/len(ious))
    
    return




            

            
      





    



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train YODA classifier')
    parser.add_argument('--model_name', type=str, default='mod2', help='model name (default: resnet18)')
    parser.add_argument('--data_dir', type=str, default='data/Kitti8/', help='path to dataset')
    parser.add_argument('--save_dir', type=str, default='images/', help='path to save images')
    parser.add_argument('--cuda', type=bool, default=True, help='enables CUDA for proces')
    parser.add_argument('--IoU', metavar='IoU_threshold', type=float, default=0.02, help='[0.02]')
    args = parser.parse_args()

    IoU_threshold = 0.02
    if args.IoU != None:
        IoU_threshold = float(args.IoU)

    print('Parameters:')
    print('\tModel name:\t', args.model_name)
    print('\tModel Path:\t', os.path.join('models', args.model_name + '.pth'))
    print('\tData dir:\t', args.data_dir)
    print('\tSave dir:\t', args.save_dir)
    print('\tIoU:\t\t', IoU_threshold)

    print('\n')

    test(args)
