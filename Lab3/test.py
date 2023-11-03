import torch
import torchvision
import torchvision.transforms as transforms
from vanilla import VanillaFrontend, VGG, ModFrontend
#from model import VGG, ModFrontend, VisualTransformerDecoder
import argparse

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

def main():
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--classes', type=int, default=100)
    parser.add_argument('--encoder', type=str, default='encoder.pth' )
    parser.add_argument('--frontend', type=str, default='frontend.pth')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device being used: {device}")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    encoder = VGG.encoder
    encoder.load_state_dict(torch.load(args.encoder, map_location=device))  
    encoder = encoder.to(device)
    model = ModFrontend(encoder, num_classes=args.classes).to(device)
    #model = VanillaFrontend(encoder, num_classes=args.classes).to(device)
    model.load_state_dict(torch.load(args.frontend, map_location=device))
    model.eval()

    top1_accuracy = 0.0
    top5_accuracy = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1_accuracy += acc1[0]
            top5_accuracy += acc5[0]
            num_batches += 1

    top1_avg_accuracy = top1_accuracy / num_batches
    top5_avg_accuracy = top5_accuracy / num_batches

    print(f'Top 1 accuracy: {top1_avg_accuracy:.2f}%, Top 5 accuracy: {top5_avg_accuracy:.2f}%')

if __name__ == "__main__":
    main()