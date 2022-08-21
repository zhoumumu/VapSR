# AttentionModule: 1-5-5
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights

class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv1(x) # 3M activation
        attn = self.conv0(attn) # 3M activation
        attn = self.conv_spatial(attn) # 3M activation
        return u * attn

class SpatialAttention(nn.Module):
    def __init__(self, d_model, d_atten):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_atten, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_atten)
        self.proj_2 = nn.Conv2d(d_atten, d_model, 1)
        #
        self.norm = nn.LayerNorm(d_model)
        #
        default_init_weights([self.norm], 0.1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut

        x = x.permute(0, 2, 3, 1) #(B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous() #(B, C, H, W)

        return x

def pixelshuffle_block(in_channels, out_channels, upscale_factor=4, kernel_size=3, stride=1):
    upconv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(2)
    upconv2 = nn.Conv2d(16, out_channels * 4, 3, 1, 1)
    lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return nn.Sequential(*[upconv1, pixel_shuffle, lrelu, upconv2, pixel_shuffle])

#both scale X2 and X3 use this version
def pixelshuffle_block_2(in_channels, out_channels, upscale_factor=2):
    upconv1 = nn.Conv2d(in_channels, 56, 3, 1, 1)
    upconv2 = nn.Conv2d(56, out_channels * upscale_factor * upscale_factor, 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return nn.Sequential(*[upconv1, lrelu, upconv2, pixel_shuffle])

def make_layer(block, n_layers, *kwargs):
    layers = []
    for _ in range(n_layers):
        layers.append(block(*kwargs))
    return nn.Sequential(*layers)

@ARCH_REGISTRY.register()
class vapsr_beta(nn.Module):

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, d_atten=64):
        super(vapsr_beta, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(SpatialAttention, num_block, num_feat, d_atten)
        #self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=2) # for VapSR-S
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # # upsample
        self.upsampler = pixelshuffle_block_2(num_feat, num_out_ch, upscale_factor=scale)


    def forward(self, feat):
        feat = self.conv_first(feat)
        body_feat = self.body(feat)

        body_out = self.conv_body(body_feat)

        feat = feat + body_out
        # upsample
        out = self.upsampler(feat)
        return out

