import torch.nn as nn

from ..utils.components import DobleConv, DownBlock, UpBlock


class DeepUNet(nn.Module):
    """
    DeepUNet: A Deep Fully Convolutional Network for Pixel-level Sea-Land Segmentation
    https://arxiv.org/abs/1709.00201
    """
    def __init__(self, input_channels, hidden_channels, n_classes, BN):
        super(DeepUNet, self).__init__()

        self.doble_conv = DobleConv(input_channels, 2 * hidden_channels, hidden_channels, BN)
        self.down_block1 = DownBlock(hidden_channels, 2 * hidden_channels, hidden_channels, BN)
        self.down_block2 = DownBlock(hidden_channels, 2 * hidden_channels, hidden_channels, BN)
        self.down_block3 = DownBlock(hidden_channels, 2 * hidden_channels, hidden_channels, BN)
        self.down_block4 = DownBlock(hidden_channels, 2 * hidden_channels, hidden_channels, BN)

        self.up_block1 = UpBlock(hidden_channels, hidden_channels, 2 * hidden_channels, hidden_channels, BN)
        self.up_block2 = UpBlock(hidden_channels, hidden_channels, 2 * hidden_channels, hidden_channels, BN)
        self.up_block3 = UpBlock(hidden_channels, hidden_channels, 2 * hidden_channels, hidden_channels, BN)
        self.up_block4 = UpBlock(hidden_channels, hidden_channels, 2 * hidden_channels, hidden_channels, BN)
        self.out = nn.Conv2d(hidden_channels, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.doble_conv(x)
        x2 = self.down_block1(x1)
        x3 = self.down_block2(x2)
        x4 = self.down_block3(x3)
        y = self.down_block4(x4)

        y = self.up_block1(y, x4)
        y = self.up_block2(y, x3)
        y = self.up_block3(y, x2)
        y = self.up_block4(y, x1)

        x, x1, x2, x3, x4 = None, None, None, None, None

        return self.out(y)
