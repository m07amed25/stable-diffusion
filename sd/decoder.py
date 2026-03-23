import torch
from torch import nn
from torch.nn import functional as F

from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))
        x += residue
        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # 0: post_quant_conv
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            # 1: conv_in
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            # 2: mid.block_1
            VAE_ResidualBlock(512, 512),
            # 3: mid.attn_1
            VAE_AttentionBlock(512),
            # 4: mid.block_2
            VAE_ResidualBlock(512, 512),
            # 5: up.3.block.0
            VAE_ResidualBlock(512, 512),
            # 6: up.3.block.1
            VAE_ResidualBlock(512, 512),
            # 7: up.3.block.2
            VAE_ResidualBlock(512, 512),
            # 8: upsample (no weights)
            nn.Upsample(scale_factor=2),
            # 9: up.3.upsample.conv
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # 10: up.2.block.0
            VAE_ResidualBlock(512, 512),
            # 11: up.2.block.1
            VAE_ResidualBlock(512, 512),
            # 12: up.2.block.2
            VAE_ResidualBlock(512, 512),
            # 13: upsample (no weights)
            nn.Upsample(scale_factor=2),
            # 14: up.2.upsample.conv
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # 15: up.1.block.0 (512 -> 256)
            VAE_ResidualBlock(512, 256),
            # 16: up.1.block.1
            VAE_ResidualBlock(256, 256),
            # 17: up.1.block.2
            VAE_ResidualBlock(256, 256),
            # 18: upsample (no weights)
            nn.Upsample(scale_factor=2),
            # 19: up.1.upsample.conv
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # 20: up.0.block.0 (256 -> 128)
            VAE_ResidualBlock(256, 128),
            # 21: up.0.block.1
            VAE_ResidualBlock(128, 128),
            # 22: up.0.block.2
            VAE_ResidualBlock(128, 128),
            # 23: norm_out
            nn.GroupNorm(32, 128),
            # 24: SiLU (no weights)
            nn.SiLU(),
            # 25: conv_out
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        # x: (Batch, 4, Height / 8, Width / 8)

        x /= 0.18215

        for module in self:
            x = module(x)

        # (Batch, 3, Height, Width)
        return x
