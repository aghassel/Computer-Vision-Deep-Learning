import torch.nn as nn
import torch.nn.functional as F
import torch


#based on https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
class Bottleneck(nn.Module):
    def __init__(self, in_channels, channels, stride=1, upsample=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride,padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.upsample = upsample
        self.upsample_ = nn.Sequential(nn.Conv2d(in_channels, channels*4,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels*4),)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        ##print(out.size())

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        ##print(out.size())
        if self.upsample:
            residual =self.upsample_(residual)

        
        ##print(out.size())
        ##print(residual.size())

        out += residual
        out = self.relu(out)

        return out


# Encoder part of the model
class ResNetEncoder(nn.Module):
    def __init__(self, input_channels=3):
        super(ResNetEncoder, self).__init__()
        self.input_channels = input_channels
        ######## resnet encoder #############
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)       
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        #conv2
        self.block1up = Bottleneck(64, 64, stride=1, upsample=True)
        layers = []
        for i in range(1,3):
            layers.append(Bottleneck(256, 64, stride=1))
        self.block1 = nn.Sequential(*layers)
        #conv3
        self.block2up = Bottleneck(256, 128, stride=2, upsample=True)
        layers = []
        for i in range(1,8):
            layers.append(Bottleneck(512, 128, stride=1))
        self.block2 = nn.Sequential(*layers)
        #conv4
        self.block3up = Bottleneck(512, 256, stride=2, upsample=True)
        layers = []
        for i in range(1,36):
            layers.append(Bottleneck(1024, 256, stride=1))
        self.block3 = nn.Sequential(*layers)
        #conv5
        self.block4up = Bottleneck(1024, 512, stride=2, upsample=True)
        layers = []
        for i in range(1,3):
            layers.append(Bottleneck(2048, 512, stride=1))
        self.block4 = nn.Sequential(*layers)
        #conv6
        self.conv6 = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(1024)

    def forward(self, x):
        #conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x2s = self.relu(x)
        x2s = self.maxpool(x2s)

        #conv2
        x2s = self.block1up(x2s)
        x2s = self.block1(x2s)

        #conv3
        x4s = self.block2up(x2s)
        x4s = self.block2(x4s)

        #conv4
        x8s = self.block3up(x4s)
        x8s = self.block3(x8s)

        #conv5
        x16s = self.block4up(x8s)
        x16s = self.block4(x16s)

        #conv6
        x32s = self.conv6(x16s)
        x32s = self.bn6(x32s)
        x32s = self.relu(x32s)

        return x32s, x16s, x8s, x4s, x2s

# Decoder part of the model
class ResNetDecoder(nn.Module):
    def __init__(self, input_channels=1024, output_channels=2):
        super(ResNetDecoder, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        ############## FCN decoder ###########
        #conv_up5
        self.conv_up5 = nn.Sequential(nn.Conv2d(input_channels, 1024, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(inplace=True))
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        #conv_up4
        self.conv_up4 = nn.Sequential(nn.Conv2d(1024+1024, 512, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True))
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        #conv_up3
        self.conv_up3 = nn.Sequential(nn.Conv2d(512+512, 256, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True))
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        #conv_up2
        self.conv_up2 = nn.Sequential(nn.Conv2d(256+256, 128, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True))
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        #conv_up1
        self.conv_up1 = nn.Sequential(nn.Conv2d(64+128, 64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        #conv7
        self.conv7 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True))
        #conv8
        self.conv8 = nn.Conv2d(32, output_channels, kernel_size=1, stride=1)

    def forward(self, x, x16s, x8s, x4s, x2s):
        #up5
        up = self.conv_up5(torch.cat((x,x16s),1))
        up = self.up5(up)

        #up4
        up = self.conv_up4(torch.cat((up,x8s),1))
        up = self.up4(up)

        #up3
        up = self.conv_up3(torch.cat((up,x4s),1))
        up = self.up3(up)

        #up2
        up = self.conv_up2(torch.cat((up,x2s),1))
        up = self.up2(up)

        #up1
        up = self.conv_up1(torch.cat((up,x),1))
        up = self.up1(up)

        #conv7
        up = self.conv7(up)

        #conv8
        out = self.conv8(up)
        seg_pred = out[:,:1,:,:]
        radial_pred = out[:,1:,:,:]

        return seg_pred, radial_pred

if __name__=="__main__":
    # Test varying input size
    import numpy as np
    import torch
    import os

    # Create an instance of the ResNetEncoder
    encoder = ResNetEncoder(input_channels=3).to('cuda')

    # Create an instance of the ResNetDecoder
    decoder = ResNetDecoder(input_channels=2048, output_channels=2).to('cuda')
    # Note: You need to specify the input_channels parameter to match the output of the encoder.

    # Generate some sample input data (adjust the shape as needed)
    batch_size = 1
    input_data = torch.randn(batch_size, 3, 480, 640).to('cuda')

    # Pass the input data through the encoder
    x32s, x16s, x8s, x4s, x2s = encoder(input_data)

    # Print the shapes of the encoder outputs
    print("Encoder Output Shapes:", x32s.shape, x16s.shape, x8s.shape, x4s.shape, x2s.shape)

    # Pass the encoder output through the decoder
    decoder_output = decoder(x32s, x16s, x8s, x4s, x2s)

    # Print the shapes of the encoder and decoder outputs

    print("Decoder Output Shape:", decoder_output[0].shape, decoder_output[1].shape)
