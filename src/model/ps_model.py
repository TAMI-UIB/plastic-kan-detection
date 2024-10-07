import math
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchmetrics.functional import spectral_angle_mapper
from torchmetrics.functional import peak_signal_noise_ratio


class ModelPansharpeningSharingOperators(nn.Module):
    def __init__(self, kernel_size, std, panchro_size, ms_size, n_resblocks, n_channels, n_iterations,
                 learn_B, learn_upsample, learn_p_tilde, learn_u_tilde, device):
        super().__init__()

        self.device = device
        self.n_channels = n_channels
        self.iterations = n_iterations

        # Low pass filter
        if learn_B:
            self.B = DobleConv(n_channels, n_channels, n_channels, BN=False)
        else:
            self.B = torchvision.transforms.GaussianBlur(kernel_size, sigma=std)

        # Basic transformations
        if learn_upsample:
            self.upsampling = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=2, stride=2)
        else:
            self.upsampling = torchvision.transforms.Resize(size=panchro_size,
                                                            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                                            antialias=None)

        self.downsampling = torchvision.transforms.Resize(size=ms_size,
                                                          interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                                                          antialias=None)

        # Composed transformations
        self.DB = torchvision.transforms.Compose([self.B, self.downsampling])
        self.DBtranspose = torchvision.transforms.Compose([self.upsampling, self.B])

        # Proximal operator defined as a ResNet
        self.prox = ProxNet(n_channels, n_channels, n_resblocks)

        # Learnable parameters
        self.gamma = nn.Parameter(torch.Tensor([1]).to(self.device))
        self.tau = nn.Parameter(torch.Tensor([0.05]).to(self.device))
        self.lmb = nn.Parameter(torch.Tensor([1000]).to(self.device))
        self.mu = nn.Parameter(torch.Tensor([10]).to(self.device))

        self.MSEloss = nn.MSELoss(reduction='mean')

        # make tildes
        self.learn_p_tilde = learn_p_tilde
        if learn_p_tilde:
            self.make_p_tilde = torchvision.transforms.Compose([
                self.B,
                self.downsampling,
                self.upsampling
            ])

        self.learn_u_tilde = learn_u_tilde
        if learn_u_tilde:
            self.make_u_tilde = self.upsampling

    def forward(self, ms, pan, p_tilde, u_tilde, **kwargs):

        u = torch.zeros_like(pan).to(self.device)
        p = torch.zeros_like(pan).to(self.device)
        q = torch.zeros_like(ms).to(self.device)

        u_barra = u.clone()

        if self.learn_p_tilde:
            p_tilde = self.make_p_tilde(pan)
        if self.learn_u_tilde:
            u_tilde = self.make_u_tilde(ms)

        for _ in range(self.iterations):
            u_anterior = u.clone()

            p = p + self.gamma * u_barra - self.gamma * self.prox(p / self.gamma + u_barra)
            q = (q + self.gamma * (self.DB(u_barra) - ms)) / (1 + self.gamma / self.mu)
            u = (u - self.tau * (p + self.DBtranspose(q) - self.lmb * p_tilde * pan * u_tilde)) / (
                    1 + self.tau * self.lmb * torch.square(p_tilde))

            u_barra = 2 * u - u_anterior

        return u



'''Auxiliar classes'''


class ProxNet(nn.Module):

    def __init__(self, in_channels, out_channels, n_blocks):
        super().__init__()
        self.blocks = n_blocks
        self.layers = nn.Sequential(*[ResBlock(in_channels, out_channels) for block in range(n_blocks)])

    def forward(self, x):
        out = self.layers(x)

        if self.blocks > 1:
            out = out + x

        return out


"""
    def get_number_params(self):

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        return params
"""


class DobleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, BN=False):
        super(DobleConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.BN = BN
        if self.BN:
            self.bn1 = nn.BatchNorm2d(mid_channels, momentum=0.5)
            self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.5)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        if self.BN:
            x = self.ReLU(self.bn1(self.conv1(x)))
            x = self.ReLU(self.bn2(self.conv2(x)))
        else:
            x = self.ReLU(self.conv1(x))
            x = self.ReLU(self.conv2(x))
        return x


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block1 = CBNReLU(in_channels, 32, 3)
        self.block2 = CBNReLU(32, 32, 3)
        self.block3 = CBNReLU(32, out_channels, 3)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        out = out + x
        return out


class CBNReLU(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_channels, out_channels, kernel_size):
        super(CBNReLU, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.5),
            nn.ReLU(inplace=False),
        )
        self.apply(self.init_xavier)

    def forward(self, x):
        x = self.layers(x)
        return x

    def init_xavier(self, module):
        if type(module) == nn.Conv2d:
            nn.init.xavier_uniform_(module.weight)


if __name__ == '__main__':
    model = ModelPansharpeningSharingOperators(9, 1.7, [20, 20], [5, 5], 1, 4, 50)
    f = torch.rand([1, 4, 5, 5])
    P = torch.rand([1, 4, 20, 20])
    P_tilde = torch.rand([1, 4, 20, 20])
    u_tilde = torch.rand([1, 4, 20, 20])

    u = model(f, P, P_tilde, u_tilde)

    print(u.shape)

    # test1 = torch.sum(model.downsample(model.gaussian_convolution(P), 4)*f)

    # test2 = torch.sum(P*model.gaussian_convolution(model.upsampling(f)))

    # print(test1-test2)
