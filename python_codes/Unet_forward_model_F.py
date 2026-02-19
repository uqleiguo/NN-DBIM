"""
Copyright (c) 2026 Lei Guo
email: l.guo3@uq.edu.au
Institution: The University of Queensland

This file is part of the NN-DBIM project.
Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import numpy as np
import torch
import torch.nn as nn


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


GT_Nx = 128
GT_Ny = 128

channels = np.array([64, 128, 256, 512, 1024])


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):

        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_ch, out_ch)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)

        return down_out, skip_out


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.double_conv = DoubleConv(in_ch, out_ch)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)

        return self.double_conv(x)


class ForwardSolver(nn.Module):
    def __init__(self, out_channels, channel_num, e_inc):
        super(ForwardSolver, self).__init__()

        self.e_inc = e_inc

        # Downsampling Path
        self.down_conv1 = DownBlock(4, channel_num[0])
        self.down_conv2 = DownBlock(channel_num[0], channel_num[1])
        self.down_conv3 = DownBlock(channel_num[1], channel_num[2])
        self.down_conv4 = DownBlock(channel_num[2], channel_num[3])

        # Bottlenect
        self.double_conv = DoubleConv(channel_num[3], channel_num[4])

        # Upsampling Path
        self.up_conv4 = UpBlock(channel_num[3] + channel_num[4], channel_num[3])
        self.up_conv3 = UpBlock(channel_num[2] + channel_num[3], channel_num[2])
        self.up_conv2 = UpBlock(channel_num[1] + channel_num[2], channel_num[1])
        self.up_conv1 = UpBlock(channel_num[0] + channel_num[1], channel_num[0])

        # Final Convolution
        self.conv_last = nn.Conv2d(channel_num[0], out_channels, kernel_size=1)
        self.conv_reg = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x, skip_out_1 = self.down_conv1(x)
        x, skip_out_2 = self.down_conv2(x)
        x, skip_out_3 = self.down_conv3(x)
        x, skip_out_4 = self.down_conv4(x)

        x = self.double_conv(x)

        x = self.up_conv4(x, skip_out_4)
        x = self.up_conv3(x, skip_out_3)
        x = self.up_conv2(x, skip_out_2)
        x = self.up_conv1(x, skip_out_1)
        x = self.conv_last(x)

        x = x + self.e_inc
        e_tot = self.conv_reg(x)

        return e_tot
