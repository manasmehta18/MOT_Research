""" Parts of the generator model """

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch.nn.modules.batchnorm import BatchNorm2d


class UpCk(nn.Module):
    """ (convolution => [BN] => ReLU) """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upck = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=2, kernel_size=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.upck(x)


class DownCk(nn.Module):
    """ (convolution => [BN] => ReLU) """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downck = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, stride=2, kernel_size=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.downck(x)


class CDk(nn.Module):
    """ (convolution => [BN] => Dropout => ReLU) """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cdk = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, stride=2, kernel_size=4),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=0.0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.cdk(x)