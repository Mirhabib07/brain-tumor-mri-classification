import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )
class UNet(nn.Module):
    def __init__(self, in_channels=1, base=16):
        super(UNet, self).__init__()
        self.enc1=double_conv(in_channels, base)
        self.pool1=nn.MaxPool2d(2)
        self.enc2=double_conv(base, base*2)
        self.pool2=nn.MaxPool2d(2)
        self.enc3=double_conv(base*2, base*4)
        self.pool3=nn.MaxPool2d(2)
        self.bottleneck=double_conv(base*4,base*8)
        self.upconv3=nn.ConvTranspose2d(base*8, base*4, kernel_size=2, stride=2)
        self.dec3=double_conv(base*8, base*4)
        self.upconv2=nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2)
        self.dec2=double_conv(base*4, base*2)
        self.upconv1=nn.ConvTranspose2d(base*2, base, kernel_size=2, stride=2)
        self.dec1=double_conv(base*2, base)
        self.out=nn.Conv2d(base, 1, 1)
    def forward(self, x):
        enc1=self.enc1(x)
        enc2=self.enc2(self.pool1(enc1))
        enc3=self.enc3(self.pool2(enc2))
        bottleneck=self.bottleneck(self.pool3(enc3))
        dec3=self.upconv3(bottleneck)
        dec3=torch.cat((dec3, enc3), dim=1)
        dec3=self.dec3(dec3)