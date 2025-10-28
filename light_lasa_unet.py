# light_lasa_unet.py

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from lasa import LASA
from se_block import SEBlock
from hybrid_decoder_block import HybridDecoderBlock
import config


class Light_LASA_Unet(nn.Module):
    def __init__(self, num_classes=2, backbone_name='mobilenet_v2', lasa_kernels=[1, 3, 5, 7]):
        super(Light_LASA_Unet, self).__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.lasa_kernels = lasa_kernels

        self._get_backbone_features(backbone_name)

        if backbone_name == 'mobilenet_v2':
            e1_ch, e2_ch, e3_ch, e4_ch, bottle_ch = 16, 24, 32, 96, 1280
            # Add SEBlocks for mobilenet as in the original implementation
            self.encoder1.add_module("SEBlock", SEBlock(e1_ch))
            self.encoder2.add_module("SEBlock", SEBlock(e2_ch))
            self.encoder3.add_module("SEBlock", SEBlock(e3_ch))
            self.encoder4.add_module("SEBlock", SEBlock(e4_ch))
        else:
            channel_info = config.BACKBONE_CHANNELS.get(backbone_name)
            if channel_info is None:
                raise ValueError(f"Channel info for '{backbone_name}' not in config.py")
            e1_ch, e2_ch, e3_ch, e4_ch, bottle_ch = \
                channel_info['e1'], channel_info['e2'], channel_info['e3'], channel_info['e4'], channel_info['bottleneck']

        self.lasa_module = LASA(in_channels=e4_ch, L_list=lasa_kernels)

        self.decoder4 = HybridDecoderBlock(bottle_ch + e4_ch, 256)
        self.aux_conv_d4 = nn.Conv2d(256, num_classes, kernel_size=1)

        self.decoder3 = HybridDecoderBlock(256 + e3_ch, 128)
        self.aux_conv_d3 = nn.Conv2d(128, num_classes, kernel_size=1)

        self.decoder2 = HybridDecoderBlock(128 + e2_ch, 64)
        self.aux_conv_d2 = nn.Conv2d(64, num_classes, kernel_size=1)

        self.decoder1 = HybridDecoderBlock(64 + e1_ch, 64)
        self.aux_conv_d1 = nn.Conv2d(64, num_classes, kernel_size=1)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _get_backbone_features(self, backbone_name):
        if backbone_name == 'mobilenet_v2':
            mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            self.encoder1 = nn.Sequential(*mobilenet.features[0:2])
            self.encoder2 = nn.Sequential(*mobilenet.features[2:4])
            self.encoder3 = nn.Sequential(*mobilenet.features[4:7])
            self.encoder4 = nn.Sequential(*mobilenet.features[7:14])
            self.bottleneck_layer = nn.Sequential(*mobilenet.features[14:])
        elif backbone_name == 'vgg19':
            features = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT).features
            self.encoder1 = nn.Sequential(*features[:6])
            self.encoder2 = nn.Sequential(*features[6:13])
            self.encoder3 = nn.Sequential(*features[13:26])
            self.encoder4 = nn.Sequential(*features[26:39])
            self.bottleneck_layer = nn.Sequential(*features[39:52])
        elif backbone_name == 'vgg16':
            features = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT).features
            # Correct slicing for vgg16_bn
            self.encoder1 = nn.Sequential(*features[:6])
            self.encoder2 = nn.Sequential(*features[6:13])
            self.encoder3 = nn.Sequential(*features[13:23])
            self.encoder4 = nn.Sequential(*features[23:33])
            self.bottleneck_layer = nn.Sequential(*features[33:43])
        else:
            raise NotImplementedError(f"Backbone '{backbone_name}' not supported.")

    def forward(self, x):
        input_h, input_w = x.shape[2:]

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4_enhanced = self.lasa_module(e4)
        bottleneck = self.bottleneck_layer(e4_enhanced)

        aux_outputs = []

        d4_input = torch.cat([F.interpolate(bottleneck, size=e4.shape[2:], mode='bilinear', align_corners=True), e4], dim=1)
        d4_out = self.decoder4(d4_input)
        aux_outputs.append(F.interpolate(self.aux_conv_d4(d4_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        d3_input = torch.cat([F.interpolate(d4_out, size=e3.shape[2:], mode='bilinear', align_corners=True), e3], dim=1)
        d3_out = self.decoder3(d3_input)
        aux_outputs.append(F.interpolate(self.aux_conv_d3(d3_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        d2_input = torch.cat([F.interpolate(d3_out, size=e2.shape[2:], mode='bilinear', align_corners=True), e2], dim=1)
        d2_out = self.decoder2(d2_input)
        aux_outputs.append(F.interpolate(self.aux_conv_d2(d2_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        d1_input = torch.cat([F.interpolate(d2_out, size=e1.shape[2:], mode='bilinear', align_corners=True), e1], dim=1)
        d1_out = self.decoder1(d1_input)
        aux_outputs.append(F.interpolate(self.aux_conv_d1(d1_out), size=(input_h, input_w), mode='bilinear', align_corners=True))

        final_output = self.final_conv(d1_out)
        final_output_upsampled = F.interpolate(final_output, size=(input_h, input_w), mode='bilinear', align_corners=True)

        return tuple(aux_outputs + [final_output_upsampled])