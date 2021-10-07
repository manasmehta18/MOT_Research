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

        self.lstm = nn.LSTM(input_size=15 * 20 * 1024, hidden_size=64, num_layers=1)
        self.linear = nn.Linear(64, 15 * 20 * 1024)

    def forward(self, x):
        x = self.encoder(x)
        x = rearrange(x, "t f h w -> t (f h w)")
        x = x.unsqueeze(dim=1)
        x = self.lstm(x)[0]
        # x is [time, 1, features]

        x = x.squeeze()
        x = self.linear(x)

        # [time, features, h, w]
        x = rearrange(x, "t (f h w)-> t f h w", f=1024, h=15, w=20)
        x = self.decoder(x)

        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 1
        self.down4 = Down(512, 1024 // factor)

        # self.down1 = DownCk(in_channels=3, out_channels=64)
        # self.down2 = DownCk(in_channels=64, out_channels=128)
        # self.down3 = DownCk(in_channels=128, out_channels=256)
        # self.down4 = DownCk(in_channels=256, out_channels=512)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        # x = self.down1(x)
        # x = self.down2(x)
        # x = self.down3(x)
        # x = self.down4(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        factor = 1
        self.up1 = Up(1024, 512 // factor, True)
        self.up2 = Up(512, 256 // factor, True)
        self.up3 = Up(256, 128 // factor, True)
        self.up4 = Up(128, 64, True)
        self.outc = OutConv(64, 1)

        # self.up1 = UpCk(in_channels=512, out_channels=256)
        # self.up2 = UpCk(in_channels=256, out_channels=128)
        # self.up3 = UpCk(in_channels=128, out_channels=64)
        # self.up4 = UpCk(in_channels=64, out_channels=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.outc(x)

        # x = self.up1(x)
        # x = self.up2(x)
        # x = self.up3(x)
        # x = self.up4(x)
        return x