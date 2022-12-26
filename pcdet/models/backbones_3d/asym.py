"""
Cylinder fork from Cylinder3D.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under Apache License [see LICENSE].
"""

import torch.nn as nn
from ...utils.spconv_utils import spconv, replace_feature
import torch
from functools import partial

from .focal_sparse_conv.focal_sparse_conv import FocalSparseConv
from .focal_sparse_conv.SemanticSeg.pyramid_ffn import PyramidFeat2D


class objDict:
    @staticmethod
    def to_object(obj: object, **data):
        obj.__dict__.update(data)


class ConfigDict:
    def __init__(self, name):
        self.name = name

    def __getitem__(self, item):
        return getattr(self, item)


class SparseSequentialBatchdict(spconv.SparseSequential):
    def __init__(self, *args, **kwargs):
        super(SparseSequentialBatchdict, self).__init__(*args, **kwargs)

    def forward(self, input, batch_dict=None):
        loss = 0
        for k, module in self._modules.items():
            if module is None:
                continue
            if isinstance(module, (FocalSparseConv,)):
                input, batch_dict, _loss = module(input, batch_dict)
                loss += _loss
            else:
                input = module(input)
        return input, batch_dict, loss


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(True),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU(True)
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                             padding=(1, 0, 0), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef2")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef4")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))
        resA = resA.replace_feature(resA.features + shortcut.features)

        return resA


class FocalConvBlock(spconv.SparseModule):
    def __init__(self, model_cfg, in_filters, out_filters, voxel_stride, kernel_size=(3, 3, 3), indice_key=None):
        super(FocalConvBlock, self).__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        use_img = model_cfg.get('USE_IMG', False)
        topk = model_cfg.get('TOPK', True)
        threshold = model_cfg.get('THRESHOLD', 0.5)
        kernel_size = model_cfg.get('KERNEL_SIZE', 3)
        mask_multi = model_cfg.get('MASK_MULTI', False)
        skip_mask_kernel = model_cfg.get('SKIP_MASK_KERNEL', False)
        skip_mask_kernel_image = model_cfg.get('SKIP_MASK_KERNEL_IMG', False)
        enlarge_voxel_channels = model_cfg.get('ENLARGE_VOXEL_CHANNELS', -1)
        img_pretrain = model_cfg.get('IMG_PRETRAIN', "../checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth")

        if use_img:
            model_cfg_seg = dict(
                name='SemDeepLabV3',
                backbone='ResNet50',
                num_class=21,  # pretrained on COCO
                args={"feat_extract_layer": ["layer1"],
                      "pretrained_path": img_pretrain},
                channel_reduce={
                    "in_channels": [256],
                    "out_channels": [16],
                    "kernel_size": [1],
                    "stride": [1],
                    "bias": [False]
                }
            )
            cfg_dict = ConfigDict('SemDeepLabV3')
            objDict.to_object(cfg_dict, **model_cfg_seg)
            self.semseg = PyramidFeat2D(optimize=True, model_cfg=cfg_dict)

            self.conv_focal_multimodal = FocalSparseConv(16, 16,
                                                         image_channel=model_cfg_seg['channel_reduce']['out_channels'][
                                                             0],
                                                         topk=topk, threshold=threshold, use_img=True,
                                                         skip_mask_kernel=skip_mask_kernel_image,
                                                         voxel_stride=1, norm_fn=norm_fn,
                                                         indice_key='spconv_focal_multimodal')

        self.special_spconv_fn = FocalSparseConv(inplanes=in_filters, planes=out_filters,
                                                 voxel_stride=voxel_stride, norm_fn=norm_fn,
                                                 mask_multi=mask_multi,
                                                 enlarge_voxel_channels=enlarge_voxel_channels,
                                                 topk=topk, threshold=threshold, kernel_size=kernel_size,
                                                 padding=kernel_size // 2,
                                                 skip_mask_kernel=skip_mask_kernel, indice_key=indice_key)
        self.use_img = use_img

    def forward(self, x, batch_dict):
        return self.special_spconv_fn(x, batch_dict)


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False, indice_key=None):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef2")
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef4")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        if pooling:
            if height_pooling:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, indice_key=indice_key, bias=False)
            else:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),
                                                padding=1, indice_key=indice_key, bias=False)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))

        resA = resA.replace_feature(resA.features + shortcut.features)

        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(UpBlock, self).__init__()
        # self.drop_out = drop_out
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key + 'up1')
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(out_filters, out_filters, indice_key=indice_key + 'up2')
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x3(out_filters, out_filters, indice_key=indice_key + 'up3')
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(out_filters)

        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        upA = self.trans_dilao(x)
        upA = upA.replace_feature(self.trans_act(upA.features))
        upA = upA.replace_feature(self.trans_bn(upA.features))

        # upsample
        upA = self.up_subm(upA)

        upA = upA.replace_feature(upA.features + skip.features)

        upE = self.conv1(upA)
        upE = upE.replace_feature(self.act1(upE.features))
        upE = upE.replace_feature(self.bn1(upE.features))

        upE = self.conv2(upE)
        upE = upE.replace_feature(self.act2(upE.features))
        upE = upE.replace_feature(self.bn2(upE.features))

        upE = self.conv3(upE)
        upE = upE.replace_feature(self.act3(upE.features))
        upE = upE.replace_feature(self.bn3(upE.features))

        return upE


class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock, self).__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters, indice_key=indice_key + "bef2")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))

        shortcut2 = self.conv1_2(x)
        shortcut2 = shortcut2.replace_feature(self.bn0_2(shortcut2.features))
        shortcut2 = shortcut2.replace_feature(self.act1_2(shortcut2.features))

        shortcut3 = self.conv1_3(x)
        shortcut3 = shortcut.replace_feature(self.bn0_3(shortcut3.features))
        shortcut3 = shortcut3.replace_feature(self.act1_3(shortcut3.features))
        shortcut = shortcut.replace_feature(shortcut.features + shortcut2.features + shortcut3.features)

        shortcut = shortcut.replace_feature(shortcut.features * x.features)

        return shortcut


class Asymm3DSpconv(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super(Asymm3DSpconv, self).__init__()
        # self.nclasses = nclasses
        self.model_cfg = model_cfg
        self.strict = False
        num_input_channels = 16
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.downCntx = ResContextBlock(input_channels, num_input_channels, indice_key="pre")
        self.resBlock2 = ResBlock(num_input_channels, 2 * num_input_channels, 0.2, height_pooling=True,
                                  indice_key="down2")
        self.resBlock3 = ResBlock(2 * num_input_channels, 4 * num_input_channels, 0.2, height_pooling=True,
                                  stride=2,
                                  indice_key="down3")
        self.resBlock4 = ResBlock(4 * num_input_channels, 8 * num_input_channels, 0.2,
                                  stride=2,
                                  height_pooling=True,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * num_input_channels, 16 * num_input_channels, 0.2,
                                  stride=2,
                                  height_pooling=True,
                                  indice_key="down5")
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(256, 256, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(256),
            nn.ReLU(),
        )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        ret = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)

        out = self.conv_out(down4b)
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': down1b,
                'x_conv2': down2b,
                'x_conv3': down3b,
                'x_conv4': down4b,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class Asymm3DSpconvFocal(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super(Asymm3DSpconvFocal, self).__init__()
        self.model_cfg = model_cfg
        self.strict = False
        self.use_img = model_cfg.USE_IMG
        if self.use_img:
            img_pretrain = model_cfg.get('IMG_PRETRAIN', "../checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth")
            model_cfg_seg = dict(
                name='SemDeepLabV3',
                backbone='ResNet50',
                num_class=21,  # pretrained on COCO
                args={"feat_extract_layer": ["layer1"],
                      "pretrained_path": img_pretrain},
                channel_reduce={
                    "in_channels": [256],
                    "out_channels": [16],
                    "kernel_size": [1],
                    "stride": [1],
                    "bias": [False]
                }
            )
            cfg_dict = ConfigDict('SemDeepLabV3')
            objDict.to_object(cfg_dict, **model_cfg_seg)
            self.semseg = PyramidFeat2D(optimize=True, model_cfg=cfg_dict)
        num_input_channels = 16
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.downCntx = ResContextBlock(input_channels, num_input_channels, indice_key="pre")
        self.resBlock2 = ResBlock(num_input_channels, 2 * num_input_channels, 0.2, height_pooling=True,
                                  indice_key="down2")
        self.spatial_focal_conv1 = FocalConvBlock(model_cfg=model_cfg, in_filters=2 * num_input_channels,
                                                  out_filters=2 * num_input_channels, voxel_stride=1,
                                                  indice_key='focal1')
        self.resBlock3 = ResBlock(2 * num_input_channels, 4 * num_input_channels, 0.2, height_pooling=True,
                                  stride=2,
                                  indice_key="down3")
        self.spatial_focal_conv2 = FocalConvBlock(model_cfg=model_cfg, in_filters=4 * num_input_channels,
                                                  out_filters=4 * num_input_channels, voxel_stride=2,
                                                  indice_key='focal2')
        self.resBlock4 = ResBlock(4 * num_input_channels, 8 * num_input_channels, 0.2,
                                  stride=2,
                                  height_pooling=True,
                                  indice_key="down4")
        self.spatial_focal_conv3 = FocalConvBlock(model_cfg=model_cfg, in_filters=8 * num_input_channels,
                                                  out_filters=8 * num_input_channels, voxel_stride=4,
                                                  indice_key='focal3')
        self.resBlock5 = ResBlock(8 * num_input_channels, 16 * num_input_channels, 0.2,
                                  stride=2,
                                  height_pooling=True,
                                  indice_key="down5")

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(256, 256, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(256),
            nn.ReLU(),
        )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256
        }
        self.forward_ret_dict = {}

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        ret = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down1c, batch_dict, loss_down1c = self.spatial_focal_conv1(down1c, batch_dict)
        down1b, batch_dict, loss_down1b = self.spatial_focal_conv1(down1b, batch_dict)
        loss_img = 0
        if self.use_img:
            x_image = self.semseg(batch_dict['images'])['layer1_feat2d']
            down1c, batch_dict, loss_img = self.conv_focal_multimodal(down1c, batch_dict, x_image)

        down2c, down2b = self.resBlock3(down1c)
        down2c, batch_dict, loss_down2c = self.spatial_focal_conv2(down2c, batch_dict)
        down2b, batch_dict, loss_down2b = self.spatial_focal_conv2(down2b, batch_dict)
        down3c, down3b = self.resBlock4(down2c)
        down3c, batch_dict, loss_down3c = self.spatial_focal_conv3(down3c, batch_dict)
        down3b, batch_dict, loss_down3b = self.spatial_focal_conv3(down3b, batch_dict)
        down4c, down4b = self.resBlock5(down3c)

        self.forward_ret_dict['loss_box_of_pts'] = loss_down1c + loss_down1b + loss_down2c + loss_down2b + +loss_down3c\
                                                + loss_down3b + loss_img
        out = self.conv_out(down4b)
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': down1b,
                'x_conv2': down2b,
                'x_conv3': down3b,
                'x_conv4': down4b,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict
