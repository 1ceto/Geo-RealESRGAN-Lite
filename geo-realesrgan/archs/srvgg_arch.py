import torch
from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn as nn
from torch.nn import functional as F

# === 新增 SPP 模块 ===
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[1, 2, 4]):
        super(SpatialPyramidPooling, self).__init__()
        self.pool_sizes = pool_sizes

    def forward(self, x):
        features = [F.adaptive_avg_pool2d(x, output_size) for output_size in self.pool_sizes]
        features = [F.interpolate(f, size=x.size()[2:], mode='bilinear', align_corners=False) for f in features]
        out = torch.cat(features + [x], dim=1)
        return out

@ARCH_REGISTRY.register()
class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # 新增：SPP 输出通道数 = (num_feat) * (len(pool_sizes) + 1)
        self.spp = SpatialPyramidPooling(pool_sizes=[1, 2, 4])
        spp_out_channels = num_feat * (len([1, 2, 4]) + 1)

        # SPP 后接卷积降维
        self.reduce_conv = nn.Conv2d(spp_out_channels, num_feat, 1, 1, 0)
        self.body.append(self.reduce_conv)

        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body) - 2):  # 最后两个是 reduce_conv + final conv
            out = self.body[i](out)
        out = self.spp(out)
        out = self.body[-2](out)  # reduce_conv
        out = self.body[-1](out)  # final conv
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out
