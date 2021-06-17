from torch import nn
from model.backbone import Resnet
import numpy as np


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                       stride=1, padding=0, dilation=1, bias=False):
        super(ConvBnRelu, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ParsingNet(nn.Module):
    def __init__(self, size=(288, 800), pretrained=True, backbone='50',
                       cls_dim=(37, 10, 4), use_aux=False):
        super(ParsingNet, self).__init__()

        self.size = size
        self.w = size[0]
        self.h = size[1]
        self.cls_dim = cls_dim  # (num_gridding, num_cls_per_lane, num_of_lanes)
        self.use_aux = use_aux
        self.total_dim = np.prod(cls_dim)

        # Input: NCHW
        # Output: (w+1) * sample_rows * 4
        self.model = Resnet(backbone, pretrained=pretrained)

        if self.use_aux:
            self.aux_header2 = nn.Sequential(
                ConvBnRelu(128, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34', '18'] \
                    else ConvBnRelu(512, 128, kernel_size=3, stride=1, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
                ConvBnRelu(128, 128, 3, padding=1)
            )
            self.aux_header3 = nn.Sequential(
                ConvBnRelu(256, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] \
                    else ConvBnRelu(1024, 128, kernel_size=3, stride=1, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
                ConvBnRelu(128, 128, 3, padding=1)
            )
            self.aux_header4 = nn.Sequential(
                ConvBnRelu(512, 128, kernel_size=3, stride=1, padding=1) if backbone in ['34','18'] \
                    else ConvBnRelu(2048, 128, kernel_size=3, stride=1, padding=1),
                ConvBnRelu(128, 128, 3, padding=1)
            )
            self.aux_combine = nn.Sequential(
                ConvBnRelu(384, 256, 3,padding=2, dilation=2),
                ConvBnRelu(256, 128, 3,padding=2, dilation=2),
                ConvBnRelu(128, 128, 3,padding=2, dilation=2),
                ConvBnRelu(128, 128, 3,padding=4, dilation=4),
                nn.Conv2d(128, cls_dim[-1] + 1, 1)
                # Output : n, num_of_lanes+1, h, w
            )
            initialize_weights(self.aux_header2, self.aux_header3, self.aux_header4, self.aux_combine)



def initialize_weights(*models):
    for model in models:
        real_init_weights(model)


def real_init_weights(m):
    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('Unknown module', m)


