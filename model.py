import torch
import torch.nn as nn
import torch.nn.functional as F

# === SPP 模块 ===
class SpatialPyramidPooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool4 = nn.AdaptiveAvgPool2d(4)
        self.conv = nn.Conv2d(in_channels * (1 + 4 + 16), in_channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()
        p1 = self.pool1(x).view(b, c, -1)
        p2 = self.pool2(x).view(b, c, -1)
        p4 = self.pool4(x).view(b, c, -1)
        out = torch.cat([p1, p2, p4], dim=2)
        out = out.view(b, -1, 1, 1)
        out = self.conv(out)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        return out

# === Dense Block ===
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_channels=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, growth_channels, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(growth_channels, in_channels, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.layers(x)

# === Residual in Residual Dense Block (RRDB) ===
class RRDB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.db1 = DenseBlock(in_channels)
        self.db2 = DenseBlock(in_channels)
        self.db3 = DenseBlock(in_channels)

    def forward(self, x):
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return x + out * 0.2

# === Generator ===
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_feat=64, num_blocks=5):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1)

        self.spp = SpatialPyramidPooling(num_feat)

        self.rrdb_blocks = nn.Sequential(*[RRDB(num_feat) for _ in range(num_blocks)])

        self.trunk_conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.upsample = nn.Sequential(
            nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(True),
            nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(True),
        )

        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)

    def forward(self, x):
        fea = self.conv_first(x)
        fea = self.spp(fea)
        trunk = self.trunk_conv(self.rrdb_blocks(fea))
        fea = fea + trunk
        fea = self.upsample(fea)
        out = self.conv_last(fea)
        return out

# === Discriminator ===
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        def conv_block(in_feat, out_feat, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, 3, stride, 1),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.encoder = nn.Sequential(
            conv_block(in_channels, 64, stride=1),
            conv_block(64, 64, stride=2),
            conv_block(64, 128, stride=1),
            conv_block(128, 128, stride=2),
            conv_block(128, 256, stride=1),
            conv_block(256, 256, stride=2),
            conv_block(256, 512, stride=1),
            conv_block(512, 512, stride=2),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=1)
        )

    def forward(self, x):
        feat = self.encoder(x)
        out = self.decoder(feat)
        return out.view(out.size(0), -1)
