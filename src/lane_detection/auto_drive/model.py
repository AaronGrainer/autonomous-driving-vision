import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import load


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        self.conv3x1_2 = nn.Conv2d(
            chann,
            chann,
            (3, 1),
            stride=1,
            padding=(1 * dilated, 0),
            bias=True,
            dilation=(dilated, 1),
        )
        self.conv1x3_2 = nn.Conv2d(
            chann,
            chann,
            (1, 3),
            stride=1,
            padding=(0, 1 * dilated),
            bias=True,
            dilation=(1, dilated),
        )
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        return F.relu(output + input)


class Encoder(nn.Module):
    def __init__(self, num_classes, dropout_1=0.03, dropout_2=0.3):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, dropout_1, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, dropout_2, 2))
            self.layers.append(non_bottleneck_1d(128, dropout_2, 4))
            self.layers.append(non_bottleneck_1d(128, dropout_2, 8))
            self.layers.append(non_bottleneck_1d(128, dropout_2, 16))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(
            16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True
        )

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


class SpatialConv(nn.Module):
    """SCNN head"""

    def __init__(self, num_channels=128):
        super().__init__()
        self.conv_d = nn.Conv2d(num_channels, num_channels, (1, 9), padding=(0, 4))
        self.conv_u = nn.Conv2d(num_channels, num_channels, (1, 9), padding=(0, 4))
        self.conv_r = nn.Conv2d(num_channels, num_channels, (9, 1), padding=(4, 0))
        self.conv_l = nn.Conv2d(num_channels, num_channels, (9, 1), padding=(4, 0))
        self._adjust_initializations(num_channels=num_channels)

    def _adjust_initializations(self, num_channels=128):
        # https://github.com/XingangPan/SCNN/issues/82
        bound = math.sqrt(2.0 / (num_channels * 9 * 5))
        nn.init.uniform_(self.conv_d.weight, -bound, bound)
        nn.init.uniform_(self.conv_u.weight, -bound, bound)
        nn.init.uniform_(self.conv_r.weight, -bound, bound)
        nn.init.uniform_(self.conv_l.weight, -bound, bound)

    def forward(self, input):
        output = input

        # First one remains unchanged (according to the original paper), why not add a relu afterwards?
        # Update and send to next
        # Down
        for i in range(1, output.shape[2]):
            output[:, :, i : i + 1, :].add_(F.relu(self.conv_d(output[:, :, i - 1 : i, :])))
        # Up
        for i in range(output.shape[2] - 2, 0, -1):
            output[:, :, i : i + 1, :].add_(F.relu(self.conv_u(output[:, :, i + 1 : i + 2, :])))
        # Right
        for i in range(1, output.shape[3]):
            output[:, :, :, i : i + 1].add_(F.relu(self.conv_r(output[:, :, :, i - 1 : i])))
        # Left
        for i in range(output.shape[3] - 2, 0, -1):
            output[:, :, :, i : i + 1].add_(F.relu(self.conv_l(output[:, :, :, i + 1 : i + 2])))

        return output


class EDLaneExist(nn.Module):
    """Lane exist head for ERFNet and ENet
    Really tricky without global pooling
    """

    def __init__(self, num_output, flattened_size=3965, dropout=0.1, pool="avg"):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Conv2d(128, 32, (3, 3), stride=1, padding=(4, 4), bias=False, dilation=(4, 4))
        )
        self.layers.append(nn.BatchNorm2d(32, eps=1e-03))

        self.layers_final = nn.ModuleList()
        self.layers_final.append(nn.Dropout2d(dropout))
        self.layers_final.append(nn.Conv2d(32, 5, (1, 1), stride=1, padding=(0, 0), bias=True))

        if pool == "max":
            self.pool = nn.MaxPool2d(2, stride=2)
        elif pool == "avg":
            self.pool = nn.AvgPool2d(2, stride=2)
        else:
            raise RuntimeError("This type of pool has not been defined yet!")

        self.linear1 = nn.Linear(flattened_size, 128)
        self.linear2 = nn.Linear(128, num_output)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)

        output = F.relu(output)

        for layer in self.layers_final:
            output = layer(output)

        output = F.softmax(output, dim=1)
        output = self.pool(output)
        output = output.flatten(start_dim=1)
        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)

        return output


class ERFNet(nn.Module):
    def __init__(
        self,
        num_classes,
        encoder=None,
        num_lanes=0,
        dropout_1=0.03,
        dropout_2=0.3,
        flattened_size=3965,
        scnn=False,
    ):
        super().__init__()
        if encoder is None:
            self.encoder = Encoder(
                num_classes=num_classes, dropout_1=dropout_1, dropout_2=dropout_2
            )
        else:
            self.encoder = encoder

        self.decoder = Decoder(num_classes)

        if scnn:
            self.spatial_conv = SpatialConv()
        else:
            self.spatial_conv = None

        if num_lanes > 0:
            self.lane_classifier = EDLaneExist(
                num_output=num_lanes, flattened_size=flattened_size, dropout=dropout_2, pool="max"
            )
        else:
            self.lane_classifier = None

    def forward(self, input, only_encode=False):
        out = OrderedDict()
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)  # predict=False by default
            if self.spatial_conv is not None:
                output = self.spatial_conv(output)
            out["out"] = self.decoder.forward(output)

            if self.lane_classifier is not None:
                out["lane"] = self.lane_classifier(output)
            return out


def erfnet_resnet(
    pretrained_weights,
    num_classes=19,
    num_lanes=0,
    dropout_1=0.1,
    dropout_2=0.1,
    flattened_size=4500,
    scnn=False,
):
    """Constructs a ERFNet model with ResNet-style backbone.

    Args:
        pretrained_weights (str): If not None, load ImageNet pre-trained weights from this filename
    """
    net = ERFNet(
        num_classes=num_classes,
        encoder=None,
        num_lanes=num_lanes,
        dropout_1=dropout_1,
        dropout_2=dropout_2,
        flattened_size=flattened_size,
        scnn=scnn,
    )
    if pretrained_weights is not None:  # Load ImageNet pre-trained weights
        saved_weights = load(pretrained_weights)["state_dict"]
        original_weights = net.state_dict()
        for key in saved_weights.keys():
            my_key = key.replace("module.features.", "")
            if my_key in original_weights.keys():
                original_weights[my_key] = saved_weights[key]
        net.load_state_dict(original_weights)
    return net
