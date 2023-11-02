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

class VGGResNet(nn.Module):
    def __init__(self, pretrained_vgg=None):
        super(VGGResNet, self).__init__()

        # Initialize VGG encoder with the option to load a pretrained model
        if pretrained_vgg:
            self.encoder = torch.load(pretrained_vgg)
        else:
            vgg_pretrained_features = vgg16(pretrained=True).features
            self.encoder = nn.Sequential(*list(vgg_pretrained_features.children())[:-1])  # remove the last maxpool layer

        # Initialize ResNet decoder
        self.decoder = nn.Sequential(
            # conv_up5
            nn.Conv2d(2048 + 1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # conv_up4
            nn.Conv2d(1024 + 512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # conv_up3
            nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # conv_up2
            nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # conv_up1
            nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        # Initialize skip connections
        self.skip1 = nn.Identity()  # For conv_up5
        self.skip2 = nn.Identity()  # For conv_up4
        self.skip3 = nn.Identity()  # For conv_up3
        self.skip4 = nn.Identity()  # For conv_up2
        self.skip5 = nn.Identity()  # For conv_up1

    def forward(self, x):
        # Forward pass through the encoder
        encoder_outputs = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in [4, 9, 16, 23, 30]:  # layer indices where skip connections are made
                encoder_outputs.append(x)

        # Initialize decoder input with the last encoder output
        decoder_input = encoder_outputs.pop()

        # Forward pass through the decoder with skip connections
        for i, layer in enumerate(self.decoder):
            if i in [1, 3, 5, 7]:  # indices where skip connections are to be concatenated
                skip_output = encoder_outputs.pop()  # get the corresponding encoder output
                decoder_input = torch.cat((decoder_input, skip_output), 1)  # concatenate along the channel dimension
            decoder_input = layer(decoder_input)

        return decoder_input

# Test the model with a 32x32x3 image tensor
if __name__ == '__main__':
    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 32, 32)

    # Instantiate the model
    model = VGGResNet()

    # Pass the dummy input through the model
    output = model(dummy_input)
    print(output.size())
