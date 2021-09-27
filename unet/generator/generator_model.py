""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from .generator_parts import *
from einops import rearrange

class Generator(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Generator, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ck1 = Ck(in_channels=3, out_channels=64)
        self.ck2 = Ck(in_channels=64, out_channels=128)
        self.ck3 = Ck(in_channels=128, out_channels=256)

    def forward(self, x):
        x = self.ck1(x)
        x = self.ck2(x)
        x = self.ck3(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ck1 = Ck(in_channels=3, out_channels=64)
        self.ck2 = Ck(in_channels=64, out_channels=128)
        self.ck3 = Ck(in_channels=128, out_channels=256)

    def forward(self, x):
        x = self.ck1(x)
        x = self.ck2(x)
        x = self.ck3(x)
        return x