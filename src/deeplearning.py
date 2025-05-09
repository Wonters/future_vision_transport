import torch
from torch import nn as nn
from torch.nn import functional as F
from torchvision import models


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 conv_layers, kernel_size=3,
                 padding='same', dilation=1,
                 bn=False, pool=False, name=''):
        super().__init__()
        layers = []
        for i in range(conv_layers):
            layers.append(nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding='same' if padding == 'same' else dilation,
                dilation=dilation,
                bias=False
            ))
            if bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DilatedNet(nn.Module):
    def __init__(self, height, width, num_classes, use_ctx_module=False, bn=True):
        super().__init__()
        self.img_height = height
        self.img_width = width
        self.nclasses = num_classes
        self.use_ctx_module = use_ctx_module

        nf = 64

        self.block1 = ConvBlock(3, nf * 1, conv_layers=2, bn=bn, pool=True, name='1')
        self.block2 = ConvBlock(nf * 1, nf * 2, conv_layers=2, bn=bn, pool=True, name='2')
        self.block3 = ConvBlock(nf * 2, nf * 4, conv_layers=3, bn=bn, pool=True, name='3')
        self.block4 = ConvBlock(nf * 4, nf * 8, conv_layers=3, bn=bn, name='4')
        self.block5 = ConvBlock(nf * 8, nf * 8, conv_layers=3, bn=bn, dilation=2, name='5')
        self.fcn1 = ConvBlock(nf * 8, nf * 64, conv_layers=1, kernel_size=7, dilation=4, bn=bn, name='FCN1')
        self.dropout1 = nn.Dropout(0.5)
        self.fcn2 = ConvBlock(nf * 64, nf * 64, conv_layers=1, kernel_size=1, bn=bn, name='FCN2')
        self.dropout2 = nn.Dropout(0.5)
        self.frontend_output = nn.Conv2d(nf * 64, self.nclasses, kernel_size=1)

        if self.use_ctx_module:
            self.ctx1 = ConvBlock(self.nclasses, self.nclasses * 2, conv_layers=2, bn=bn, name='ctx1')
            self.ctx2 = ConvBlock(self.nclasses * 2, self.nclasses * 4, conv_layers=1, dilation=2, bn=bn, name='ctx2')
            self.ctx3 = ConvBlock(self.nclasses * 4, self.nclasses * 8, conv_layers=1, dilation=4, bn=bn, name='ctx3')
            self.ctx4 = ConvBlock(self.nclasses * 8, self.nclasses * 16, conv_layers=1, dilation=8, bn=bn, name='ctx4')
            self.ctx5 = ConvBlock(self.nclasses * 16, self.nclasses * 32, conv_layers=1, dilation=16, bn=bn, name='ctx5')
            self.ctx7 = ConvBlock(self.nclasses * 32, self.nclasses * 32, conv_layers=1, bn=bn, name='ctx7')
            self.ctx_output = nn.Conv2d(self.nclasses * 32, self.nclasses, kernel_size=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.fcn1(x)
        x = self.dropout1(x)
        x = self.fcn2(x)
        x = self.dropout2(x)
        x = self.frontend_output(x)

        if self.use_ctx_module:
            x = self.ctx1(x)
            x = self.ctx2(x)
            x = self.ctx3(x)
            x = self.ctx4(x)
            x = self.ctx5(x)
            x = self.ctx7(x)
            x = self.ctx_output(x)

        # Bilinear upsampling
        x = F.interpolate(x, size=(self.img_height, self.img_width), mode='bilinear', align_corners=False)
        #x = F.softmax(x, dim=1)
        return x


class SegmentedVGG16(nn.Module):
    """
    Do segmentation in label mask shape (B,C,H,W) channel first
    """
    def __init__(self, width, height, num_classes: int =8):
        super().__init__()
        self.input_width = width
        self.input_height = height
        self.num_classes = num_classes
        self.model = models.vgg16(pretrained=True)
        self.features = self.model.features
        # 512 to 8 classes
        self.classifier = nn.Conv2d(512, self.num_classes, kernel_size=1)


    def forward(self, x):
        """"""
        x = self.features(x)
        x = self.classifier(x)
        # from cats number to size of the image, Upsampling brutal
        x = F.interpolate(x, size=(self.input_height, self.input_width), mode='bilinear', align_corners=False)
        # Transpose from class card to pixel card
        #x = F.softmax(x, dim=1)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=8):
        super().__init__()

        # Encoder
        self.enc1 = self.contracting_block(in_channels, 64)
        self.enc2 = self.contracting_block(64, 128)
        self.enc3 = self.contracting_block(128, 256)
        self.enc4 = self.contracting_block(256, 512)

        # Bottleneck
        self.bottleneck = self.contracting_block(512, 1024)

        # Decoder
        self.upconv4 = self.expansive_block(1024, 512)
        self.upconv3 = self.expansive_block(512 + 512, 256)
        self.upconv2 = self.expansive_block(256 + 256, 128)
        self.upconv1 = self.expansive_block(128 + 128, 64)

        # Final output layer
        self.final = nn.Conv2d(64 + 64, out_channels, kernel_size=1)

    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        return block

    def expansive_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        return block

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)

        out = self.final(d1)
        out = F.interpolate(out, size=(1024, 2048), mode='bilinear', align_corners=False)
        return out