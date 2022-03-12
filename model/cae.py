import torch
import torch.nn as nn
import random
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv= nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            double_conv(in_ch, out_ch),
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class ReconCAE(nn.Module):
    def __init__(self, features_root=32, tot_frame_num=5, use_flow=True):
        super(ReconCAE, self).__init__()
        self.raw_channel_num = 3  # RGB channel no.
        self.of_channel_num = 2  # optical flow channel no.
        self.tot_frame_num = tot_frame_num
        self.features_root = features_root
        self.use_flow = use_flow

        in_channels = self.raw_channel_num * tot_frame_num
        raw_out_channels = in_channels
        of_out_channels = self.of_channel_num * tot_frame_num

        # raw AE
        self.raw_ae = nn.Sequential(inconv(in_channels, features_root),
                                    down(features_root, features_root * 2),
                                    down(features_root * 2, features_root * 4),
                                    down(features_root * 4, features_root * 8),
                                    up(features_root * 8, features_root * 4),
                                    up(features_root * 4, features_root * 2),
                                    up(features_root * 2, features_root),
                                    outconv(features_root, raw_out_channels))

        if self.use_flow:
            self.of_ae = nn.Sequential(inconv(in_channels, features_root),
                                       down(features_root, features_root * 2),
                                       down(features_root * 2, features_root * 4),
                                       down(features_root * 4, features_root * 8),
                                       up(features_root * 8, features_root * 4),
                                       up(features_root * 4, features_root * 2),
                                       up(features_root * 2, features_root),
                                       outconv(features_root, of_out_channels))

    def forward(self, raw, of):
        target_raw = raw
        if self.use_flow:
            target_of = of
        else:
            target_of = torch.zeros(1).cuda().type(torch.cuda.FloatTensor)

        output_raw = self.raw_ae(raw)
        if self.use_flow:
            output_of = self.of_ae(raw)
        else:
            output_of = torch.zeros(1).cuda().type(torch.cuda.FloatTensor)

        return target_raw, target_of, output_raw, output_of
