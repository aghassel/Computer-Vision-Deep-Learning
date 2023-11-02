import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

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

class VanillaFrontend(nn.Module):
    def __init__(self, encoder, num_classes=100):
        super(VanillaFrontend, self).__init__()

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

class VisualTransformerDecoder(nn.Module):
    def __init__(self, encoder, channel_size, num_classes):
        super(VisualTransformerDecoder, self).__init__()
        self.encoder = encoder
        self.channel_size = channel_size
        self.num_classes = num_classes
        self.transformer_decoder = nn.Sequential(
            nn.Linear(512, self.channel_size),
            nn.LayerNorm(self.channel_size),
            nn.TransformerDecoder(
                nn.TransformerDecoderLayer(d_model=self.channel_size, nhead=8),
                num_layers=6
            ),
            nn.Linear(self.channel_size, self.num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
        x = self.transformer_decoder(x)
        return x

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = self.fc(x)
        return x * y.expand_as(x)

class ModFrontend(nn.Module):
    def __init__(self, encoder, num_classes=100):
        super(ModFrontend, self).__init__()

        self.encoder = encoder

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.intermediate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            SqueezeExcitationBlock(1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(512, num_classes)
        self.match_dimension = nn.Linear(512, 512)

    def forward(self, x):

        encoder_output = self.encoder(x)
        pooled_output = self.pool(encoder_output)
        residual = self.match_dimension(pooled_output.view(pooled_output.size(0), -1))
        x = self.intermediate(pooled_output)
        x = x + residual

        return self.classifier(x)
