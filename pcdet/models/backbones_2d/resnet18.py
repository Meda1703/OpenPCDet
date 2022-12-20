import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


"""
Resnet18 - layers [2, 2, 2, 2]
"""


class ResNet(nn.Module):

    def __init__(self, model_cfg, input_channels, zero_init_residual=True):
        super(ResNet, self).__init__()

        self.model_cfg = model_cfg
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        layers = self.model_cfg.LAYERS
        self.inplanes = input_channels
        self.num_filters = model_cfg.NUM_FILTERS
        self.layer1 = self._make_layer(BasicBlock, self.num_filters[0], layers[0], stride=2)
        self.layer2 = self._make_layer(BasicBlock, self.num_filters[1], layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, self.num_filters[2], layers[2], stride=2)
        self.num_bev_features = self.model_cfg.UPSAMPLE_FILTERS

        self.up = nn.Upsample(scale_factor=4, mode='bilinear',
                              align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(
                self.num_filters[2]+self.num_filters[0],
                self.num_bev_features,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(self.num_bev_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.num_bev_features,
                self.num_bev_features,
                kernel_size=3,
                padding=1,
                bias=False),
            nn.BatchNorm2d(self.num_bev_features),
            nn.ReLU(inplace=True),
        )

        # self.conv = nn.Sequential(
        #     nn.Conv2d(self.num_filters[2]+self.num_filters[1],
        #               self.num_filters[2]+self.num_filters[1],
        #               kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(self.num_filters[2]+self.num_filters[1]),
        #     nn.ReLU(inplace=True)
        # )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(self.model_cfg.UPSAMPLE_FILTERS,
                      self.model_cfg.UPSAMPLE_FILTERS,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.model_cfg.UPSAMPLE_FILTERS),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.model_cfg.UPSAMPLE_FILTERS, self.model_cfg.UPSAMPLE_FILTERS, kernel_size=1, padding=0),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, data_dict):
        # 2 x 256 x 200 x 176
        spatial_features = data_dict['spatial_features']
        # backbone
        x = spatial_features
        x = self.layer1(x)
        x1 = x
        x = self.layer2(x)

        x = self.layer3(x)
        # neck
        x = self.up(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv(x)
        # 2 x 256 x 200 x 176
        x = self.up2(x)
        data_dict['spatial_features_2d'] = x
        return data_dict
