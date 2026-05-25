"""DCGAN-style generator and discriminator/critic for 64x64 RGB images."""

import torch
from torch import nn


class Generator(nn.Module):
    """Map z in R^128 to a 64x64 RGB image."""

    def __init__(self, z_dim: int = 128, image_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            # z: (N, z_dim, 1, 1) -> (N, 512, 4, 4)
            nn.ConvTranspose2d(z_dim, base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(True),
            # -> (N, 256, 8, 8)
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            # -> (N, 128, 16, 16)
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            # -> (N, 64, 32, 32)
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            # -> (N, 3, 64, 64), normalized image range [-1, 1]
            nn.ConvTranspose2d(base_channels, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)


class Discriminator(nn.Module):
    """DCGAN discriminator/critic.

    The final scalar has no sigmoid. Vanilla GAN feeds it to BCEWithLogitsLoss,
    LSGAN treats it as a least-squares score, and WGAN treats it as a critic score.
    """

    def __init__(self, image_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # (N, 3, 64, 64) -> (N, 64, 32, 32)
            nn.Conv2d(image_channels, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (N, 128, 16, 16)
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (N, 256, 8, 8)
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (N, 512, 4, 4)
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (N, 1, 1, 1)
            nn.Conv2d(base_channels * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image).view(-1)


def initialize_dcgan_weights(module: nn.Module) -> None:
    """DCGAN initialization: conv weights N(0, 0.02), BN weights N(1, 0.02)."""
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)

