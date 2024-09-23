import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop


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


class UNet(nn.Module):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/abs/1505.04597
    """
    def __init__(self, channels, hidden_channels, n_classes=1, BN=True):
        super(UNet, self).__init__()
        self.doble_conv = DobleConv(channels, hidden_channels, hidden_channels, BN)

        self.Encoder1 = Encoder(hidden_channels, hidden_channels * 2, hidden_channels * 2, BN)
        self.Encoder2 = Encoder(hidden_channels * 2, hidden_channels * 4, hidden_channels * 4, BN)
        self.Encoder3 = Encoder(hidden_channels * 4, hidden_channels * 8, hidden_channels * 8, BN)
        self.Encoder4 = Encoder(hidden_channels * 8, hidden_channels * 16, hidden_channels * 16, BN)

        self.Decoder1 = Decoder(hidden_channels * 16, hidden_channels * 8, hidden_channels * 8, hidden_channels * 8, BN)
        self.Decoder2 = Decoder(hidden_channels * 8, hidden_channels * 4, hidden_channels * 4, hidden_channels * 4, BN)
        self.Decoder3 = Decoder(hidden_channels * 4, hidden_channels * 2, hidden_channels * 2, hidden_channels * 2, BN)
        self.Decoder4 = Decoder(hidden_channels * 2, hidden_channels, hidden_channels, hidden_channels, BN)
        self.out = nn.Conv2d(hidden_channels, n_classes, kernel_size=1)
        self.ReLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # def forward(self, x):
    #     x1 = self.doble_conv(x)
    #     x2 = self.Encoder1(x1)
    #     x3 = self.Encoder2(x2)
    #     x4 = self.Encoder3(x3)
    #     x = self.Encoder4(x4)
    #
    #     x = self.Decoder1(x, x4)
    #     x = self.Decoder2(x, x3)
    #     x = self.Decoder3(x, x2)
    #     x = self.Decoder4(x, x1)
    #     x = self.out(x)
    #
    #     return x
    def forward(self, x):
       return torch.mean(x[:, -2:, :, :], keepdim=True)