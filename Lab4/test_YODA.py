from model import ModFrontend, VGG
import os
import torch
from torchvision import datasets, transforms, models
from Kitti.KittiDataset import KittiDataset
from Kitti.KittiAnchors import Anchors
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm

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

    count = 0
    for item in tqdm(enumerate(dataset), total=len(dataset), desc="Testing", unit="images", leave=False):
        idx, (image, label) = item
        original_image_size = image.shape[:2]
        idx = dataset.class_label['Car']
        car_ROIs = dataset.strip_ROIs(class_ID=idx, label_list=label)
        ROIs, boxes = anchors.get_anchor_ROIs(image, anchors.calc_anchor_centers(image.shape, anchors.grid), anchors.shapes)

        img_copy = image.copy()
        transformed_ROIs = []

        roi_counts = 0
        for ROI in ROIs:
            ROI = Image.fromarray(ROI)
            ROI = transform(ROI)
            roi_counts += 1
            transformed_ROIs.append(ROI)
    

        if roi_counts > 50:
            batch1 = transformed_ROIs[:75]
            batch2 = transformed_ROIs[75:]

            ROIs_tensor_batch1 = torch.stack(batch1).to(device)
            ROIs_tensor_batch2 = torch.stack(batch2).to(device)

            outputs_batch1 = model(ROIs_tensor_batch1).squeeze()
            outputs_batch2 = model(ROIs_tensor_batch2).squeeze()

            outputs = torch.cat((outputs_batch1, outputs_batch2))
        else:
            ROIs_tensor = torch.stack(transformed_ROIs).to(device)
            outputs = model(ROIs_tensor).squeeze()

        outputs = torch.sigmoid(outputs)
        outputs = torch.round(outputs)

        # Rest of the code...

        ROIs_tensor = torch.stack(transformed_ROIs).to(device)

        outputs = model(ROIs_tensor).squeeze()
        outputs = torch.sigmoid(outputs)
        outputs = torch.round(outputs)

        for j in range(len(boxes)):
            box = boxes[j]
            pred = outputs[j].item()
            if pred == 1:
                pt1 = box[0]
                pt2 = box[1]
                pt1 = (int(pt1[1]), int(pt1[0]))
                pt2 = (int(pt2[1]), int(pt2[0]))
                cv2.rectangle(img_copy, pt1, pt2, (0, 255, 255))

                iou = anchors.calc_max_IoU(box, car_ROIs)
                ious.append(iou)
                cv2.putText(img_copy, f"{iou:.2f}", (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if args.save_dir and args.save_imgs:
            save_path = os.path.join(args.save_dir, f"processed_{count}.png")  # Changed from idx to count
            cv2.imwrite(save_path, img_copy)
        count += 1

    # Calculate and print mean IoU
    mean_iou = sum(ious) / len(ious) if ious else 0
    print(f"Mean IoU: {mean_iou}")

    return



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train YODA classifier')
    parser.add_argument('--model_name', type=str, default='mod2', help='model name (default: resnet18)')
    parser.add_argument('--data_dir', type=str, default='data/Kitti8/', help='path to dataset')
    parser.add_argument('--save_imgs', type=bool, default=False, help='save images with ROIs drawn on them')
    parser.add_argument('--save_dir', type=str, default='images/', help='path to save images')
    parser.add_argument('--cuda', type=bool, default=False, help='enables CUDA for proces')
    args = parser.parse_args()


    print('Parameters:')
    print('\tModel name:\t', args.model_name)
    print('\tModel Path:\t', os.path.join('models', args.model_name + '.pth'))
    print('\tData dir:\t', args.data_dir)
    print('\tSave dir:\t', args.save_dir)

    print('\n')

    test(args)
