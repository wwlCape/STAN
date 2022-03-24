### Spatio-Temporal Alignment Network
import torch
from math import sqrt
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from networks.FAC.kernelconv2d import KernelConv2D
from networks.submodules import *
from networks import submodules

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class TransformNet(nn.Module):
    def __init__(self, opts, conv=submodules.default_conv):
        super(TransformNet, self).__init__()
        ks_2d = 3
        ch3 = 64
        n_resgroups = opts.n_resgroups
        n_resblocks = opts.n_resblocks
        n_feats = opts.n_feats
        kernel_size = 3
        reduction = opts.reduction 
        scale = 4
        self.gamma = nn.Parameter(torch.ones(1))
        act = nn.ReLU(True)
        # define head module
        modules_head = [conv(21, n_feats, kernel_size), conv(n_feats, n_feats, kernel_size),resnet_block(n_feats, kernel_size=kernel_size),resnet_block(n_feats, kernel_size=kernel_size)]
        # define body module
        self.RG = nn.ModuleList([ResidualGroup(conv, n_feats, kernel_size, reduction, \
                                              act=act, res_scale=opts.res_scale, n_resblocks=n_resblocks) for _ in range(n_resgroups)])
        self.conv_last = conv(n_feats, n_feats, kernel_size)
        # define tail module
        modules_tail = [conv(n_feats, n_feats, kernel_size),
            submodules.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, opts.n_colors, kernel_size)]
        self.head = nn.Sequential(*modules_head)
        self.tail = nn.Sequential(*modules_tail)
        self.fea = nn.Conv2d(2*ch3, ch3, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=1)
        self.epoch = 0
        self.kconv_warp = KernelConv2D.KernelConv2D(kernel_size=ks_2d)
        self.fac_warp = nn.Sequential(
            conv1(ch3, ch3, kernel_size=kernel_size),
            resnet_block(ch3, kernel_size=kernel_size),
            resnet_block(ch3, kernel_size=kernel_size),
            conv1(ch3, ch3 * ks_2d ** 2, kernel_size=1))

        self.kconv4 = conv1(ch3 * ks_2d ** 2, ch3, kernel_size=1)
        self.ex = conv(opts.n_colors , ch3, kernel_size)
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)

        return nn.ModuleList(layers)

    def forward(self, F1,F2,F3,F4,F5,F6,F7,output_last_fea):
        merge = torch.cat([F1,F2,F3,F4,F5,F6,F7], 1)
        x = self.head(merge) 
        residual = x
        center_feature = self.ex(F4)
        ###Filter Adaptive Alignment Module (FAAM)
        kernel_warp_1 = self.fac_warp(x)
        if output_last_fea is None:
            output_last_fea = torch.cat([residual, center_feature],1) 
        output_last_fea = self.fea(output_last_fea)
        conv_a_k_1 = self.kconv_warp(output_last_fea, kernel_warp_1)
        kernel_warp_2 = self.fac_warp(conv_a_k_1)
        conv_a_k_2 = self.kconv_warp(output_last_fea, kernel_warp_2)
        kernel_warp_3 = self.fac_warp(conv_a_k_2)
        conv_a_k_3 = self.kconv_warp(output_last_fea, kernel_warp_3)
       
        aligned_cat = torch.cat([center_feature, conv_a_k_3],1)

        fusion= self.fea(aligned_cat)
        share_source = fusion
        ###Share Source Skip Connection
        for i,l in enumerate(self.RG):
            fusion = l(fusion) + self.gamma*share_source
        out = self.conv_last(fusion)
        out += share_source
        ###Up-sample
        sr = self.tail(out)
        base = F.interpolate(F4, scale_factor=4, mode='bilinear', align_corners=False)
        sr = sr + base
        return sr, aligned_cat
