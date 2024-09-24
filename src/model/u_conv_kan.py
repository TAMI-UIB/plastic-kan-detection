import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop


class RadialBasisFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 4,
            denominator: float = None,
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class FastKANConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, input_dim, output_dim, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0, **norm_kwargs):
        super(FastKANConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.grid_size = grid_size
        self.base_activation = base_activation()
        self.grid_range = grid_range
        self.norm_kwargs = norm_kwargs

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        self.base_conv = nn.ModuleList([conv_class(input_dim // groups,
                                                   output_dim // groups,
                                                   kernel_size,
                                                   stride,
                                                   padding,
                                                   dilation,
                                                   groups=1,
                                                   bias=False) for _ in range(groups)])

        self.spline_conv = nn.ModuleList([conv_class(grid_size * input_dim // groups,
                                                     output_dim // groups,
                                                     kernel_size,
                                                     stride,
                                                     padding,
                                                     dilation,
                                                     groups=1,
                                                     bias=False) for _ in range(groups)])

        self.layer_norm = nn.ModuleList([norm_class(input_dim // groups, **norm_kwargs) for _ in range(groups)])

        self.rbf = RadialBasisFunction(grid_range[0], grid_range[1], grid_size)

        self.dropout = None
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)

        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

        for conv_layer in self.spline_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

    def forward_fast_kan(self, x, group_index):

        # Apply base activation to input and then linear transform with base weights
        base_output = self.base_conv[group_index](self.base_activation(x))
        if self.dropout is not None:
            x = self.dropout(x)
        spline_basis = self.rbf(self.layer_norm[group_index](x))
        spline_basis = spline_basis.moveaxis(-1, 2).flatten(1, 2)
        spline_output = self.spline_conv[group_index](spline_basis)
        x = base_output + spline_output

        return x

    def forward(self, x):
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_fast_kan(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class FastKANConv2DLayer(FastKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0,
                 norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super(FastKANConv2DLayer, self).__init__(nn.Conv2d, norm_layer,
                                                 input_dim, output_dim,
                                                 kernel_size,
                                                 groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                 ndim=2,
                                                 grid_size=grid_size, base_activation=base_activation,
                                                 grid_range=grid_range,
                                                 dropout=dropout, **norm_kwargs)


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

        self.conv1 = FastKANConv2DLayer(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = FastKANConv2DLayer(mid_channels, out_channels, kernel_size=3, padding=1)
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


class UConvKAN(nn.Module):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/abs/1505.04597
    """
    def __init__(self, channels, hidden_channels, n_classes=1, BN=True):
        super(UConvKAN, self).__init__()
        self.doble_conv = DobleConv(channels, hidden_channels, hidden_channels, BN)

        self.Encoder1 = Encoder(hidden_channels, hidden_channels * 2, hidden_channels * 2, BN)
        self.Encoder2 = Encoder(hidden_channels * 2, hidden_channels * 4, hidden_channels * 4, BN)
        self.Encoder3 = Encoder(hidden_channels * 4, hidden_channels * 8, hidden_channels * 8, BN)
        self.Encoder4 = Encoder(hidden_channels * 8, hidden_channels * 16, hidden_channels * 16, BN)

        self.Decoder1 = Decoder(hidden_channels * 16, hidden_channels * 8, hidden_channels * 8, hidden_channels * 8, BN)
        self.Decoder2 = Decoder(hidden_channels * 8, hidden_channels * 4, hidden_channels * 4, hidden_channels * 4, BN)
        self.Decoder3 = Decoder(hidden_channels * 4, hidden_channels * 2, hidden_channels * 2, hidden_channels * 2, BN)
        self.Decoder4 = Decoder(hidden_channels * 2, hidden_channels, hidden_channels, hidden_channels, BN)
        self.out = FastKANConv2DLayer(hidden_channels, n_classes, kernel_size=1)
        self.ReLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.doble_conv(x)
        x2 = self.Encoder1(x1)
        x3 = self.Encoder2(x2)
        x4 = self.Encoder3(x3)
        x = self.Encoder4(x4)

        x = self.Decoder1(x, x4)
        x = self.Decoder2(x, x3)
        x = self.Decoder3(x, x2)
        x = self.Decoder4(x, x1)
        x = self.out(x)

        return x