""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from .generator_parts import *
from einops import rearrange

class Generator(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Generator, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = UpCk(in_channels=3, out_channels=64)
        self.up2 = UpCk(in_channels=64, out_channels=128)
        self.up3 = UpCk(in_channels=128, out_channels=256)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DownCk(in_channels=256, out_channels=128)
        self.down2 = DownCk(in_channels=128, out_channels=64)
        self.down3 = DownCk(in_channels=64, out_channels=1)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x