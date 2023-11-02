#   Adapted from:
#       https://github.com/naoto0804/pytorch-AdaIN

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
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
    )
        
class VanillaFrontend(nn.Module):
    def __init__(self, encoder, num_classes=100):
        super(VanillaFrontend, self).__init__()

        self.encoder = encoder  

        self.frontend = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),   # Convert the 512 feature maps to 1x1x512
            nn.Flatten(),                   # Flatten the 1x1x512 to 512x1
            nn.Linear(512, num_classes)  
        )

    def forward(self, x):
        
        with torch.no_grad():
            x = self.encoder(x)  

        return self.frontend(x)

class ModifiedFrontend(nn.Module):
    def __init__(self, encoder, num_classes=100):
        super(ModifiedFrontend, self).__init__()

        self.encoder = encoder
        # Load a pre-trained ResNeXt-50 model
        resnext = models.resnext50_32x4d(pretrained=True)

        # Remove the fully connected layer
        modules = list(resnext.children())[:-1]
        self.encoder = nn.Sequential(*modules)

        # Replace the frontend with your desired structure
        self.frontend = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Convert the feature maps to 1x1
            nn.Flatten(),                 # Flatten the 1x1 feature map
            nn.Linear(2048, num_classes) # Fully connected layer
        )

    def forward(self, x):
        x = self.encoder(x)
        
        return self.frontend(x)

class CustomFrontend(nn.Module):
    def __init__(self, num_classes):
        super(CustomFrontend, self).__init__()
        self.resnet_frontend = models.resnet50(pretrained=True)  # Load a pretrained ResNet model
        # Modify the ResNet model if needed (e.g., remove the last layer)
        num_features = self.resnet_frontend.fc.in_features
        self.resnet_frontend.fc = nn.Linear(num_features, num_classes)  # Adjust input size for ResNet variant

    def forward(self, x):
        x = self.encoder
        return self.resnet_frontend(x)


class VisualTransformerDecoder(nn.Module):
    def __init__(self, encoder, channel_size, num_classes):
        super(VisualTransformerDecoder, self).__init__()
        self.encoder = encoder
        self.channel_size = channel_size
        self.num_classes = num_classes
        self.linear = nn.Linear(512 * 4 * 4, self.channel_size)
        self.norm = nn.LayerNorm(self.channel_size)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.channel_size, nhead=8),
            num_layers=6
        )
        self.output_layer = nn.Linear(self.channel_size, self.num_classes)
    
    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the output of the encoder
        
        # Manually apply the layers that were previously in nn.Sequential
        x = self.linear(x)
        x = self.norm(x)
        
        # The transformer decoder can now be called with 'tgt' and 'memory' as expected.
        # Assuming that the 'memory' should be the same as 'tgt' here, which might not be the case.
        # You will need to adjust based on your model's specifics.
        memory = x
        x = self.transformer_decoder(tgt=x, memory=memory)
        
        x = self.output_layer(x)
        return x
