import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop

######################################
#### Blocks for UNet and deepUNet ####
######################################


#### BASE ####
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

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
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


#### UNET ####
class Encoder(nn.Module):
    """
    Enconder block, in the down path:
    Maxpool2d -> DobleConv
    """
    def __init__(self, in_channels, mid_channels, out_channels, BN):
        super(Encoder, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.doble_conv = DobleConv(in_channels, mid_channels, out_channels, BN)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.doble_conv(x)
        return x


class Decoder(nn.Module):
    """
    Decoder block, in the up path:
    UpConv & concat -> DobleConv
    """
    def __init__(self, in_channels, mid_channels1, mid_channels2, out_channels, BN):
        super(Decoder, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, mid_channels1, kernel_size=2, stride=2)
        self.doble_conv = DobleConv(mid_channels1 * 2, mid_channels2, out_channels, BN)

    def forward(self, x, conc_layer):
        x = self.up(x)
        conc_layer = CenterCrop(size=(x.size()[2], x.size()[3]))(conc_layer)
        x = torch.cat([x, conc_layer], dim=1)
        x = self.doble_conv(x)
        return x


#### DEEPUNET ####
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


#### DEEPUNET FE ####
class Decoder_residual(nn.Module):
    """
    Modified version of the Decoder from the UNet, sum instead of concat:
    UpConv & sum -> DobleConv
    """
    def __init__(self, in_channels, mid_channels, out_channels, BN):
        super(Decoder_residual, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.doble_conv = DobleConv(in_channels, mid_channels, out_channels, BN)

    def forward(self, x, sum_layer):
        x = self.up(x)
        sum_layer = CenterCrop(size=(x.size()[2], x.size()[3]))(sum_layer)
        x = x + sum_layer
        x = self.doble_conv(x)
        return x


class FeatureExtractor_conc(nn.Module):
    """
    Shallow UNet-like model
    """
    def __init__(self, in_channels, out_channels, BN):
        super(FeatureExtractor_conc, self).__init__()

        self.doble_conv = DobleConv(in_channels, in_channels * 2, in_channels * 2, BN)
        self.Encoder1 = Encoder(in_channels * 2, in_channels * 4, in_channels * 4, BN)
        self.Encoder2 = Encoder(in_channels * 4, in_channels * 8, in_channels * 8, BN)

        self.Decoder1 = Decoder(in_channels * 8, in_channels * 4, in_channels * 4, in_channels * 4, BN)
        self.Decoder2 = Decoder(in_channels * 4, in_channels * 2, in_channels * 2, in_channels * 2, BN)
        self.out = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):

        x1 = self.doble_conv(x)
        x2 = self.Encoder1(x1)
        x = self.Encoder2(x2)

        x = self.Decoder1(x, x2)
        x = self.Decoder2(x, x1)
        x = self.out(x)

        return x


class FeatureExtractor_sum(nn.Module):
    """
    Shallow UNet-like model with sum instead of conc
    """
    def __init__(self, in_channels, out_channels, BN):
        super(FeatureExtractor_sum, self).__init__()

        self.doble_conv = DobleConv(in_channels, in_channels * 2, in_channels * 2, BN)
        self.Encoder1 = Encoder(in_channels * 2, in_channels * 4, in_channels * 4, BN)
        self.Encoder2 = Encoder(in_channels * 4, in_channels * 8, in_channels * 8, BN)
        self.Encoder3 = Encoder(in_channels * 8, in_channels * 8, in_channels * 8, BN)

        self.Decoder1 = Decoder_residual(in_channels * 8, in_channels * 8, in_channels * 4, BN)
        self.Decoder2 = Decoder_residual(in_channels * 4, in_channels * 4, in_channels * 2, BN)
        self.Decoder3 = Decoder_residual(in_channels * 2, in_channels * 2, in_channels, BN)
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.doble_conv(x)
        x2 = self.Encoder1(x1)
        x3 = self.Encoder2(x2)
        x = self.Encoder3(x3)

        x = self.Decoder1(x, x3)
        x = self.Decoder2(x, x2)
        x = self.Decoder3(x, x1)
        x = self.out(x)

        return x