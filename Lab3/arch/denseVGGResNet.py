import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

class VGG:
    encoder = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),
    )

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        out = self.relu(out)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
            ))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            x = layer(torch.cat(features, dim=1))
            features.append(x)
        return torch.cat(features, dim=1)

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=1, stride=1)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class DenseResNet152(nn.Module):
    def __init__(self, encoder, num_classes=100):
        super(DenseResNet152, self).__init__()
        self.encoder = encoder
        self.features = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block1 = DenseBlock(in_channels=64, growth_rate=32, num_layers=6)
        self.trans1 = TransitionBlock(in_channels=256, out_channels=128)
        self.block2 = DenseBlock(in_channels=128, growth_rate=32, num_layers=12)
        self.trans2 = TransitionBlock(in_channels=512, out_channels=256)
        self.block3 = DenseBlock(in_channels=256, growth_rate=32, num_layers=36)
        self.trans3 = TransitionBlock(in_channels=1408, out_channels=512)
        self.block4 = DenseBlock(in_channels=512, growth_rate=32, num_layers=8)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.features(x)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)
        x = self.trans3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        

# Test the model with a 32x32x3 image tensor
if __name__ == '__main__':
    # Create a dummy input tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate the model
    dummy_input = torch.randn(2, 3, 32, 32).to(device)
    encoder = VGG.encoder
    encoder.load_state_dict(torch.load('../models/encoder.pth', map_location=device))  
    encoder = encoder.to(device)
    model = DenseResNet152(encoder, num_classes=10).to(device)  # Change the number of classes to match your classification task
    
    # Pass the dummy input through the model
    output = model(dummy_input)
    print(output.size())