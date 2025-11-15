import torch.nn.functional as F
from layers import *
import numpy as np
import pywt


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))


def dwt_init(x):
    _, _, h, w = x.shape
    if h % 2 != 0:
        x = F.pad(x, (0, 0, 0, 1))
    if w % 2 != 0:
        x = F.pad(x, (0, 1, 0, 0))

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, x_HL + x_LH + x_HH


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False
    def forward(self, x):
        return dwt_init(x)


class DWT_transform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv1x1_low = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1x1_high = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    def forward(self, x):
        dwt_low_frequency, dwt_high_frequency = self.dwt(x)
        dwt_low_frequency = self.conv1x1_low(dwt_low_frequency)
        dwt_high_frequency = self.conv1x1_high(dwt_high_frequency)
        return dwt_low_frequency, dwt_high_frequency


class SimpleLowFrequencyEnhance(nn.Module):
    def __init__(self, dim):
        super(SimpleLowFrequencyEnhance, self).__init__()
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pointwise_conv = nn.Conv2d(dim, dim, kernel_size=1)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        attention = self.channel_attention(x)
        x_out = x * attention
        return x_out


class tre_feature_fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.low_attmap = SimpleLowFrequencyEnhance(dim)
        self.high_attmap = Bottle(dim, dim)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, low_frequency, high_frequency, encode):
        low_frequency = self.low_attmap(low_frequency)
        high_frequency = self.high_attmap(high_frequency)
        fusion = self.alpha * low_frequency + (1 - self.alpha) * high_frequency
        return encode + fusion


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        sa_map = self.conv(concat)
        return x * self.sigmoid(sa_map)


class DSConv(nn.Module):
   

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.gelu(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class DifferenceDiscriminationModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        gn_half = min(4, channels // 2)
        gn_full = min(4, channels)

        self.x_path = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.GroupNorm(gn_half, channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 3, padding=1),
        )

        self.y_path = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.GroupNorm(gn_half, channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 3, padding=1),
        )

        self.ms_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
            nn.GroupNorm(gn_full, channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def shift_tensor(self, tensor, direction='down'):
        shifted_tensor = torch.zeros_like(tensor)
        if direction == 'down':
            shifted_tensor[:, :, 1:, :] = tensor[:, :, :-1, :]
        elif direction == 'right':
            shifted_tensor[:, :, :, 1:] = tensor[:, :, :, :-1]
        else:
            raise ValueError("Direction must be 'down' or 'right'.")
        return shifted_tensor

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input contains NaN or Inf values!")

        ms_feat = self.ms_conv(x)

        shifted_right = self.shift_tensor(ms_feat, 'right')
        shifted_down = self.shift_tensor(ms_feat, 'down')

        diff_x = torch.abs(ms_feat - shifted_right)
        diff_y = torch.abs(ms_feat - shifted_down)

        diff_x = torch.clamp(diff_x, max=1e1)
        diff_y = torch.clamp(diff_y, max=1e1)

        scale_factor = 0.1
        diff_x_enhanced = self.x_path(diff_x) * (1 + scale_factor * torch.sigmoid(diff_x))
        diff_y_enhanced = self.y_path(diff_y) * (1 + scale_factor * torch.sigmoid(diff_y))

        diff_x_enhanced = torch.clamp(diff_x_enhanced, min=-1e2, max=1e2)
        diff_y_enhanced = torch.clamp(diff_y_enhanced, min=-1e2, max=1e2)

        return diff_x_enhanced, diff_y_enhanced


class Bottle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = in_channels // 2

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.ddm = DifferenceDiscriminationModule(out_channels)

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        diff_x, diff_y = self.ddm(out)

        fused = out + diff_x + diff_y
        fused = self.fusion(fused)

        return fused + residual


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last=False, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x


class UConvV3(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.dyt_e1 = DynamicTanh(normalized_shape=[in_channels], channels_last=False)
        self.dyt_d1 = DynamicTanh(normalized_shape=[in_channels], channels_last=False)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 5, padding=4, dilation=2),
            nn.BatchNorm2d(in_channels),
            DSConv(in_channels, in_channels)
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
        )

        self.encoder2 = nn.Sequential(
            ChannelAttention(in_channels),
            DSConv(in_channels, in_channels),
            SpatialAttention()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU()
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, 3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU(),
            ChannelAttention(in_channels * 2),
            SpatialAttention()
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.decoder2 = nn.Sequential(
            DSConv(in_channels * 2, in_channels),
            ChannelAttention(in_channels)
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.decoder1 = nn.Sequential(
            DSConv(in_channels * 2, in_channels),
            SpatialAttention()
        )

        self.final = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1)
        )

    def _align_tensors(self, source, target):
        """Align source tensor size to match target"""
        if source.shape[2:] != target.shape[2:]:
            source = F.interpolate(
                source,
                size=target.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        return source

    def forward(self, x):
        e1 = self.encoder1[0](x)
        e1 = self.encoder1[1](e1)
        e1 = self.dyt_e1(e1)
        e1 = self.encoder1[2](e1)

        d1 = self.down1[0](e1)
        d1 = self.down1[1](d1)
        d1 = self.dyt_d1(d1)

        e2 = self.encoder2(d1)
        d2 = self.down2(e2)

        bn = self.bottleneck(d2)

        u2 = self.up2(bn)
        u2 = self._align_tensors(u2, e2)
        u2 = torch.cat([u2, e2], dim=1)
        d2_out = self.decoder2(u2)

        u1 = self.up1(d2_out)
        u1 = self._align_tensors(u1, e1)
        u1 = torch.cat([u1, e1], dim=1)
        d1_out = self.decoder1(u1)

        out = self.final(d1_out)
        return out + x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [UConvV3(dim) for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class FreqFusionNet(nn.Module):
    def __init__(self, num_res=4):
        super(FreqFusionNet, self).__init__()

        base_channel = 32

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)
        self.dwt = DWT_transform(in_channels=32, out_channels=64)
        self.dwt2 = DWT_transform(in_channels=64, out_channels=128)
        self.fusion1 = tre_feature_fusion(dim=128)
        self.fusion2 = tre_feature_fusion(dim=64)

        self.layer1 = BasicLayer(dim=32, depth=2)
        self.layer2 = BasicLayer(dim=64, depth=2)
        self.layer3 = BasicLayer(dim=128, depth=2)
        self.layer4 = BasicLayer(dim=128, depth=2)
        self.layer5 = BasicLayer(dim=64, depth=2)
        self.layer6 = BasicLayer(dim=32, depth=2)
        
    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()
        
        x_ = self.feat_extract[0](x)
        res1 = self.layer1(x_)

        l_0, h_0 = self.dwt(res1)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)

        res2 = self.layer2(z)
        res2 = self.fusion2(l_0, h_0, res2)

        l, h = self.dwt2(res2)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.layer3(z)

        l = F.interpolate(l, size=(z.shape[2], z.shape[3]))
        z = self.fusion1(l, h, z)
        z = self.layer4(z)
        z_ = self.ConvsOut[0](z)

        z = self.feat_extract[3](z)

        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.layer5(z)
        z_ = self.ConvsOut[1](z)
        
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.layer6(z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


def build_net():

    return FreqFusionNet()
