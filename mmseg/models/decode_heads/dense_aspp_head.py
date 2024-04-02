# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.utils import resize
from mmseg.models.decode_heads.aspp_head import ASPPHead


# class DepthwiseSeparableASPPModule(ASPPModule):
#     """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
#     conv."""

    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #     for i, dilation in enumerate(self.dilations):
    #         if dilation > 1:
    #             self[i] = DepthwiseSeparableConvModule(
    #                 self.in_channels,
    #                 self.channels,
    #                 3,
    #                 dilation=dilation,
    #                 padding=dilation,
    #                 norm_cfg=self.norm_cfg,
    #                 act_cfg=self.act_cfg)


@MODELS.register_module()
class DenseASPPHead(ASPPHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.

    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.

    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
        c1_in_channels是浅层特征的输入通道数,即low_level_channels
        in_channels是主干部分的输入通道数
        channels是主干特征经过aspp处理后的cat前的通道数256,
        c1_channels是浅层特征经过处理之后cat前的通道数48
    """

    def __init__(self, c1_in_channels, c1_channels, **kwargs):
        super().__init__(**kwargs)
        assert c1_in_channels >= 0
        #aspp特征提取模块
        #self.aspp_modules = DepthwiseSeparableASPPModule(
        #    dilations=self.dilations,
         #   in_channels=self.in_channels,
         #   channels=self.channels,
         #  conv_cfg=self.conv_cfg,
         #   norm_cfg=self.norm_cfg,
         #   act_cfg=self.act_cfg)
        self.denseaspp = _DenseASPPBlock(self.in_channels, 
                                         512, 256, norm_layer=nn.BatchNorm2d, norm_kwargs=None)
        #self.aspp_modules = _DenseASPPBlock(
        #    in_channels=self.in_channels,
         #   inter_channels1=512,
         #   inter_channels2=256,
         #   )
        #浅层特征边：1x1 conv通道数调整
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        #输出主干特征
        # aspp_outs.extend(self.denseaspp(x))
        aspp_outs.append(self.denseaspp(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        #输出浅层特征c1_out,
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            #对主干特征上采样
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            #将浅层特征和主干特征cat
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        #3x3卷积输出
        output = self.cls_seg(output)
        return output
    
class StripPooling(nn.Module):
    def __init__(self, in_channels, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))  # 1*W
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))  # H*1
        inter_channels = int(in_channels / 4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.conv4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(True))
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 1, bias=False),
                                   nn.BatchNorm2d(in_channels))
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1(x)
        x2 = F.interpolate(self.conv2(self.pool1(x1)), (h, w), **self._up_kwargs)  # 结构图的1*W的部分
        x3 = F.interpolate(self.conv3(self.pool2(x1)), (h, w), **self._up_kwargs)  # 结构图的H*1的部分
        x4 = self.conv4(F.relu_(x2 + x3))  # 结合1*W和H*1的特征
        out = self.conv5(x4)
        return F.relu_(x + out)  # 将输出的特征与原始输入特征结合

class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPConv, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1)),
        self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3, dilation=atrous_rate, padding=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features


class _DenseASPPBlock(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPBlock, self).__init__()
        self.aspp_3 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 3, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 6, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 12, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 18, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 24, 0.1,
                                      norm_layer, norm_kwargs)
        self.SP = StripPooling(320, up_kwargs={'mode': 'bilinear', 'align_corners': True})

    def forward(self, x):
        x1 = self.SP(x)
        aspp3 = self.aspp_3(x)

        x = torch.cat([aspp3, x], dim=1)

        aspp6 = self.aspp_6(x)
        x = torch.cat([aspp6, x], dim=1)

        aspp12 = self.aspp_12(x)
        x = torch.cat([aspp12, x], dim=1)

        aspp18 = self.aspp_18(x)
        x = torch.cat([aspp18, x], dim=1)

        aspp24 = self.aspp_24(x)
        x = torch.cat([aspp24, x], dim=1)
        x = torch.cat([x, x1], dim=1)

        return x