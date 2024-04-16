# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.utils import resize
from mmseg.models.decode_heads.aspp_head import ASPPHead
from torch.nn.parameter import Parameter

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
class GAUSPASPPHead(ASPPHead):
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
        self.aspp = ASPP(in_channels=self.in_channels, out_channels=256)
        self.gau = GAU(256, 48)
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
                c1_channels,
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
        # 这个卷积和3，2，1的卷积核需要背下来，一个不改变分辨率，一个降低未原来一半
        self.comp_conv = nn.Conv2d(in_channels=2432, out_channels=2560, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)

        #输出主干特征
        # aspp_outs.extend(self.denseaspp(x))
        aspp_outs = self.aspp(x)
        
        output = self.bottleneck(aspp_outs) # in:[4, 2432, 16, 16] ->
        #输出浅层特征c1_out,

        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = self.gau(output, c1_output)
        #     gfu = GlobalFeatureUpsample1(low_channels=48,out_channels=48)
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # c1_output = gfu(c1_output).to(device)
            # self.conv1 = conv_block(512, 512, kernel_size=1, stride=1, padding=0, use_bn_act=True)
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # output = self.conv1(output).to(device)
            #output = self.conv1(output)
            # output = torch.cat([output, c1_output], dim=1)
        #     #对主干特征上采样
        #     output = resize(
        #         input=output,
        #         size=c1_output.shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)
        #     #将浅层特征和主干特征cat
        #     output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        # #3x3卷积输出
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
    
class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        # Global Attention Upsample
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)
        self.conv = nn.Conv2d(channels_low, channels_low, kernel_size=1, padding=0, bias=False)
        if upsample:
            self.conv_upsample = nn.ConvTranspose2d(channels_high, channels_low, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.comp_conv = nn.Conv2d(in_channels=304, out_channels=256, kernel_size=3, stride=1, padding=1)


    def forward(self, fms_high, fms_low, fm_mask=None):
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :param fm_mask:
        :return: fms_att_upsample
        """
        [b, c, h, w] = fms_high.size()

        # fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        # fms_high_gp = self.conv1x1(fms_high_gp)
        # fms_high_gp = self.bn_high(fms_high_gp)
        # fms_high_gp = self.relu(fms_high_gp)

        # # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        '''GAU模块'''
        fms_low_mask = self.conv3x3(fms_low)
        fms_cat=torch.cat((fms_low_mask,fms_high),dim=1)
        fms_cat=nn.AvgPool2d(fms_cat.shape[2:])(fms_high).view(len(fms_cat), c, 1, 1)
        fms_cat=self.conv1x1(fms_cat)
        fms_cat=self.LeakyReLU(fms_cat)
        fms_cat=self.conv(fms_cat)
        fms_cat=torch.sigmoid(fms_cat)

        # fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_cat * fms_low_mask
        if self.upsample:
            # out = self.relu(
                out1 = torch.cat((fms_att, fms_high),dim=1)
                out1 = self.comp_conv(out1)
                out1 = self.conv_upsample(out1)
                
                out3 = self.bn_upsample(out1)
                out = self.relu(out3)
                
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)

        return out
