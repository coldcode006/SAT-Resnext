import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def get_inplanes():
    return [128, 256, 512, 1024]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class sat_layer3d(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, input_size, reduction=16, groups=64):
        super(sa_layer3d, self).__init__()
        self.groups = groups

        self.n, self.c, self.d, self.h, self.w = input_size.shape

        self.avg_pool1 = nn.AdaptiveAvgPool3d((self.d, 1, 1))
        self.avg_pool2 = nn.AdaptiveAvgPool3d((1, self.h, self.w))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), self.d, 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), self.d, 1, 1))
        # self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1,1, 1))
        # self.sbias = Parameter(torch.ones(1, channel // (2 * groups),1, 1, 1))
        self.sigmoid = nn.Sigmoid()
        # self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))
        self.conv1 = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, d, h, w = x.shape

        x = x.reshape(b, groups, -1, d, h, w)
        x = x.permute(0, 2, 1, 3, 4, 5)

        # flatten
        x = x.reshape(b, -1, d, h, w)

        return x

    def forward(self, x):
        b, c, d, h, w = x.shape

        x = x.reshape(b * self.groups, -1, d, h, w)
        x_0, x_1 = x.chunk(2, dim=1)
        #         b, c,_,d, h, w = x_0.shape

        # avg_pool1 = nn.AdaptiveAvgPool3d((d,1,1))
        # avg_pool2 = nn.AdaptiveAvgPool3d((1,h,w))
        # channel attention
        xn = self.avg_pool1(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)
        # max_out=self.max_pool(x_0).view([b,c])
        # # max_out = self.fc(self.max_pool(x_0))
        # # avg_out = self.fc(self.avg_pool(x_0))
        # avg_out=self.avg_pool(x_0).view([b,c])
        # max_fc_out=self.fc(max_out)
        # avg_fc_out=self.fc(avg_out)
        # channel_out = self.sigmoid(max_fc_out + avg_fc_out)
        # xn = channel_out * x_0

        # spatial attention
        xs_1 = torch.mean(x_1, dim=1, keepdim=True)
        xs_2 = torch.max(x_1, dim=1, keepdim=True)[0]
        xs = torch.cat([xs_1, xs_2], dim=1)

        # xs=self.avg_pool2 (xs)
        xs = self.conv1(xs)

        xs = self.sigmoid(xs)
        x_out = xs.expand_as(x_1) * x_1
        # concatenate along channel axis
        out = torch.cat([xn, x_out], dim=1)
        out = out.reshape(b, -1, d, h, w)

        out = self.channel_shuffle(out, 2)
        return out
