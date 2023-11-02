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

class VGG_w_skip(nn.Module):
    def __init__(self, encoder, num_classes=100):
        super(VGG_w_skip, self).__init__()

        self.encoder = encoder

        self.frontend = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)

        return self.frontend(x)

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


class VGGResNet(nn.Module):
    def __init__(self, pretrained_vgg=None, num_classes=1000):
        super(VGGResNet, self).__init__()

        if pretrained_vgg:
            self.encoder = torch.load(pretrained_vgg)
        else:
            vgg_pretrained_features = vgg16(pretrained=True).features
            self.encoder = nn.Sequential(*list(vgg_pretrained_features.children())[:-1])

        self.decoder = nn.Sequential(
            ResBlock(512, 256, stride=2),
            ResBlock(256, 128, stride=2),
            ResBlock(128, 64, stride=2),
            ResBlock(64, 32, stride=2),
        )

        self.classifier = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        encoder_outputs = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in [4, 9, 16, 23, 30]:
                encoder_outputs.append(x)

        decoder_input = encoder_outputs.pop()

        for i, layer in enumerate(self.decoder):
            if isinstance(layer, ResBlock):
                decoder_input += encoder_outputs.pop()
            decoder_input = layer(decoder_input)

        decoder_input = decoder_input.view(decoder_input.size(0), -1)
        output = self.classifier(decoder_input)

        return output

# Test the model with a 32x32x3 image tensor
if __name__ == '__main__':
    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # Instantiate the model
    model = VGGResNet(num_classes=10)  # Change the number of classes to match your classification task
    
    # Pass the dummy input through the model
    output = model(dummy_input)
    print(output.size())