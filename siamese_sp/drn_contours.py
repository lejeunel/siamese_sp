from __future__ import print_function, division

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from siamese_sp import drn
from siamese_sp.drn import BasicBlock


class ConvBlock(nn.Module):
    def __init__(self, in_planes,
                 out_planes,
                 dilation,
                 with_coordconv=False,
                 with_coordconv_r=False):
        super(ConvBlock, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes

        self.downsample = nn.Sequential(
            nn.Conv2d(
                self.in_planes, self.out_planes,
                kernel_size=1,
                stride=1,
                bias=False),
            nn.BatchNorm2d(out_planes))
        self.convblock = BasicBlock(
            self.in_planes,
            self.out_planes,
            dilation=(dilation, dilation),
            downsample=self.downsample)

    def forward(self, x):

        if(self.addcoords is not None):
            x = self.addcoords(x)

        x = self.convblock(x)
        return x

class UpConvBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(UpConvBlock, self).__init__()

        self.inplanes = inplanes

        self.upconv1 = nn.ConvTranspose2d(
            inplanes,
            planes,
            4,
            stride=2,
            padding=1,
            output_padding=0,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                inplanes, planes, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes))

    def forward(self, x):
        residual = x

        x = self.upconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        residual = self.upsample(residual)
        x = self.relu(x + residual)
        return x


class DRNContours(nn.Module):
    def __init__(self,
                 model_name='drn_d_22',
                 pretrained=True):
        super(DRNContours, self).__init__()

        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.upconv1 = UpConvBlock(model.out_dim, model.out_dim * 2)
        self.upconv2 = UpConvBlock(model.out_dim * 2, model.out_dim * 4)
        self.upconv3 = UpConvBlock(model.out_dim * 4, model.out_dim * 8)

        modules_to_init = [
            self.upconv1, self.upconv2, self.upconv3, 
        ]
        for m in modules_to_init:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _create_convblock(self, in_planes, out_planes, dilation):
        downsample = nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_planes))
        convblock = BasicBlock(
            in_planes,
            out_planes,
            dilation=(dilation, dilation),
            downsample=downsample)
        return convblock

    def forward(self, x):
        x = self.base(x)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        
        return x
