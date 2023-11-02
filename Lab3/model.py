import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AttentionModule(nn.Module):
    # A simple self-attention module
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        attention = torch.bmm(proj_query, proj_key)
        attention = F.softmax(attention, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        out = self.gamma*out + x
        return out
    
class ModFrontend(nn.Module):
    def __init__(self, encoder, num_classes=100):
        super(ModFrontend, self).__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.intermediate = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Increase dropout rate
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Add another dropout here
        )
        # Consider using a more complex attention mechanism if needed
        self.attention = AttentionModule(512)  # Advanced attention module if applicable
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.intermediate(x)
        x = x.view(x.size(0), 512, 1, 1)
        x = self.attention(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
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