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

class ViTDecoder(nn.Module):
    def __init__(self, encoder, num_classes=100, image_size=224, patch_size=16, num_heads=8, num_layers=12, hidden_dim=768, mlp_dim=3072):
        super(ViTDecoder, self).__init__()

        self.encoder = encoder

        # Create a new Vision Transformer model without pretrained weights
        vit_model = models.VisionTransformer(
            img_size=image_size,
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_classes=0,  # No classification head
            qkv_bias=True,
            norm_layer=nn.LayerNorm
        )

        self.vit_encoder = vit_model

        # Add a custom classification head
        self.frontend = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)  # Fully connected layer
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)

        # Pass through the Vision Transformer
        x = self.vit_encoder(x)

        # Classify using the frontend
        return self.frontend(x)



