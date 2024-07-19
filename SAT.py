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
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), self.d, 1, 1)
        self.sigmoid = nn.Sigmoid()
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

        avg_pool1 = nn.AdaptiveAvgPool3d((d,1,1))
        avg_pool2 = nn.AdaptiveAvgPool3d((1,h,w))
        # channel attention
        xn = self.avg_pool1(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)


        # spatial attention
        xs=self.avg_pool2(x_1)
        xs = self.gn(xs)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, x_out], dim=1)
        out = out.reshape(b, -1, d, h, w)

        out = self.channel_shuffle(out, 2)
        return out
class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 input_size,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 cardinality=32,
                 widen_factor=1.0,
                 n_classes=101):
        #block = partialclass(block, cardinality=cardinality)
        # #super().__init__(block, layers, block_inplanes, n_input_channels,
        #                  conv1_t_size, conv1_t_stride, no_max_pool,
        #                  shortcut_type, n_classes)
        super().__init__()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.conv1 = nn.Conv3d(n_input_channels,
                               out_channels=64 ,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, block_inplanes[0],
                                       layers[0],
                                       cardinality,
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       cardinality,
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       cardinality,
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       cardinality,
                                       shortcut_type,
                                       stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(cardinality * 32 * block.expansion, n_classes)
        self.sat1 = sat_layer3d(256,input_size[0])
        self.sat2 = sat_layer3d(512,input_size[1])
        self.sat3 = sat_layer3d(1024,input_size[2])
        self.sat4 = sat_layer3d(2048,input_size[3])
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out
    
    def _make_layer(self, block, planes, blocks,cardinality,shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  cardinality=cardinality,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(in_planes=self.in_planes, planes=planes,cardinality=cardinality))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.sat1(x)
        x = self.layer2(x)
        x = self.sat2(x)
        x = self.layer3(x)
        x = self.sat3(x)
        x = self.layer4(x)
        x = self.sat4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
def generate_model_resnext(model_depth, **kwargs):
    assert model_depth in [50, 101, 152, 200]
    x1=torch.randn([4,256,8,28,28])
    x2=torch.randn([4,256,4,14,14])
    x3=torch.randn([4,256,2,7,7])
    x4=torch.randn([4,256,1,4,4])
    input_size=[x1,x2,x3,x4]
    if model_depth == 50:
        model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], get_inplanes(),input_size,
                        **kwargs)
    elif model_depth == 101:
        model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], get_inplanes(),input_size,
                        **kwargs)
    elif model_depth == 152:
        model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], get_inplanes(),input_size,
                        **kwargs)
    elif model_depth == 200:
        model = ResNeXt(ResNeXtBottleneck, [3, 24, 36, 3], get_inplanes(),input_size,
                        **kwargs)

    return model

