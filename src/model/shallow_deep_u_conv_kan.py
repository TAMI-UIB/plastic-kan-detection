import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop

from .components.kan import FastKANConv2DLayer as KANConv2D

class DobleConv(nn.Module):
    """
    Double convolution block:
    (Conv2d -> BN/IN -> ReLU) x 2
    """
    def __init__(self, in_channels, mid_channels, out_channels, BN=False):
        """
        :param in_channels: int
            input channels of the first Conv2d (number of channels of input tensor)
        :param mid_channels: int
            output channels of the first Conv2d and input channels of the second Conv2d
        :param out_channels: int
            output channels of the second Conv2d (number of channels of output tensor)
        :param BN: bool | str
            if True: batch normalization layers are applied after each Conv2d
            if False: additional layers not applied
            if 'IN': instance normalization layers are applied after each Conv2d
        """
        super(DobleConv, self).__init__()

        self.conv1 = KANConv2D(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = KANConv2D(mid_channels, out_channels, kernel_size=3, padding=1)
        self.BN = BN
        if self.BN:
            self.bn1 = nn.BatchNorm2d(mid_channels, momentum=0.01)
            self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.01)
        if self.BN == 'IN':
            self.bn1 = nn.InstanceNorm2d(mid_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        if self.BN or self.BN == 'IN':
            x = self.ReLU(self.bn1(self.conv1(x)))
            x = self.ReLU(self.bn2(self.conv2(x)))
        else:
            x = self.ReLU(self.conv1(x))
            x = self.ReLU(self.conv2(x))
        return x


class DownBlock(nn.Module):
    """
    DownBlock, in the down path:
    Maxpool2d -> DobleConv & sum
    """
    def __init__(self, in_channels, mid_channels, out_channels, BN, kernel_size=2):
        super(DownBlock, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size)
        self.doble_conv = DobleConv(in_channels, mid_channels, out_channels, BN)

    def forward(self, x):
        x = self.maxpool(x)
        return self.doble_conv(x) + x


class UpBlock(nn.Module):
    """
    UpBlock, in the up path:
    UpConv & concat -> DobleConv & sum
    """
    def __init__(self, in_channels, mid_channels1, mid_channels2, out_channels, BN, kernel_size=2):
        super(UpBlock, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, mid_channels1, kernel_size=kernel_size, stride=kernel_size)
        self.doble_conv = DobleConv(mid_channels1 * 2, mid_channels2, out_channels, BN)

    def forward(self, x, conc_layer):
        x1 = self.up(x)
        conc_layer = CenterCrop(size=(x1.size()[2], x1.size()[3]))(conc_layer)
        x = torch.cat([x1, conc_layer], dim=1)
        return self.doble_conv(x) + x1



class ShallowDeepUConvKAN(nn.Module):
    """

    """
    def __init__(self, channels, hidden_channels, n_classes=1, BN=True):
        super(ShallowDeepUConvKAN, self).__init__()

        self.doble_conv = DobleConv(channels, 2 * hidden_channels, hidden_channels, BN)
        self.down_block1 = DownBlock(hidden_channels, 2 * hidden_channels, hidden_channels, BN, kernel_size=4)
        self.down_block2 = DownBlock(hidden_channels, 2 * hidden_channels, hidden_channels, BN, kernel_size=4)
        # self.down_block3 = DownBlock(hidden_channels, 2 * hidden_channels, hidden_channels, BN)
        # self.down_block4 = DownBlock(hidden_channels, 2 * hidden_channels, hidden_channels, BN)

        self.up_block1 = UpBlock(hidden_channels, hidden_channels, 2 * hidden_channels, hidden_channels, BN, kernel_size=4)
        self.up_block2 = UpBlock(hidden_channels, hidden_channels, 2 * hidden_channels, hidden_channels, BN, kernel_size=4)
        # self.up_block3 = UpBlock(hidden_channels, hidden_channels, 2 * hidden_channels, hidden_channels, BN)
        # self.up_block4 = UpBlock(hidden_channels, hidden_channels, 2 * hidden_channels, hidden_channels, BN)
        self.out = KANConv2D(hidden_channels, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.doble_conv(x)
        x2 = self.down_block1(x1)
        y = self.down_block2(x2)
        # x4 = self.down_block3(x3)
        # y = self.down_block4(x4)

        y = self.up_block1(y, x2)
        y = self.up_block2(y, x1)
        # y = self.up_block3(y, x2)
        # y = self.up_block4(y, x1)

        x, x1, x2, x3, x4 = None, None, None, None, None

        return self.out(y)
