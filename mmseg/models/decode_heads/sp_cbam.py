# Copyright (c) OpenMMLab. All rights reserved.
from torch.nn.parameter import Parameter
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
class CBAM_SPASPPHead(ASPPHead):
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

    def __init__(self, c1_in_channels, c1_channels,  **kwargs):
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
        self.aspp = ASPP(in_channels=self.in_channels, out_channels=256)
        # self.SA = sa_layer(48)
        # self.SA1 = sa_layer(96)
        self.ca = ChannelAttention(in_planes=48)
        self.ca1 = ChannelAttention(in_planes=96)
        self.sa = SpatialAttention()
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
        self.c2_bottleneck = ConvModule(
                32,
                48,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                448,
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
        self.c3_bottleneck = ConvModule(
                96,
                96,
                3,
                dilation=2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        # 这个卷积和3，2，1的卷积核需要背下来，一个不改变分辨率，一个降低未原来一半
        self.comp_conv = nn.Conv2d(in_channels=2432, out_channels=2560, kernel_size=3, stride=1, padding=1)
    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)

        #输出主干特征
        aspp_outs = self.aspp(x)
        output = self.bottleneck(aspp_outs) # in:[4, 2432, 16, 16] ->
        #输出浅层特征c1_out,
        c2_output =self.c2_bottleneck(inputs[1])
        # c2_output = self.SA(c2_output)
        c2_output = self.ca(c2_output)*c2_output
        c2_output = self.sa(c2_output)*c2_output
        c3_output = self.c3_bottleneck(inputs[2])
        c3_output = self.ca1(c3_output)*c3_output
        c3_output = self.sa(c3_output)*c3_output
        c3_output = resize(
                input=c3_output,
                size=c2_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        c2_output = torch.cat([c3_output, c2_output], dim=1)
        

        output = resize(
                input=output,
                size=c2_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        output = torch.cat([output, c2_output], dim=1)
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

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(out_channels, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.SP = StripPooling(320, up_kwargs={'mode': 'bilinear', 'align_corners': True})
    def forward(self, x):
        [b, c, row, col] = x.size()
        # -----------------------------------------#
        #   一共五个分支
        # -----------------------------------------#
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # -----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        # -----------------------------------------#
        global_feature = torch.mean(x, 2, True) # dim=2 : 沿列取平均，即转为一行
        global_feature = torch.mean(global_feature, 3, True) # dim=3 : 沿行取平均，即转为一列
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        x1 = self.SP(x)
        # -----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        # -----------------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature,x,x1], dim=1)
        # result = self.conv_cat(feature_cat)
        return feature_cat
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# class sa_layer(nn.Module):
#     """Constructs a Channel Spatial Group module.

#     Args:
#         k_size: Adaptive selection of kernel size
#     """

#     def __init__(self, channel, groups=64):
#         super(sa_layer, self).__init__()
#         self.groups = groups
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
#         self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
#         self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
#         self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

#         self.sigmoid = nn.Sigmoid()
#         self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

#     @staticmethod
#     def channel_shuffle(x, groups):
#         b, c, h, w = x.shape

#         x = x.reshape(b, groups, -1, h, w)
#         x = x.permute(0, 2, 1, 3, 4)

#         # flatten
#         x = x.reshape(b, -1, h, w)

#         return x

#     def forward(self, x):
#         b, c, h, w = x.shape

#         x = x.reshape(b * self.groups, -1, h, w)
#         x_0, x_1 = x.chunk(2, dim=1)

#         # channel attention
#         xn = self.avg_pool(x_0)
#         xn = self.cweight * xn + self.cbias
#         xn = x_0 * self.sigmoid(xn)

#         # spatial attention
#         xs = self.gn(x_1)
#         xs = self.sweight * xs + self.sbias
#         xs = x_1 * self.sigmoid(xs)

#         # concatenate along channel axis
#         out = torch.cat([xn, xs], dim=1)
#         out = out.reshape(b, -1, h, w)

#         out = self.channel_shuffle(out, 2)
#         return out
