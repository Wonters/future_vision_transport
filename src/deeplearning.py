import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class FocalLoss(nn.Module):
    """
    Focal loss
    """
    def __init__(self, gamma=2, weight=None, ignore_index=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.gamma = gamma
    def forward(self, x, target):
        logp = -self.ce(x, target)
        p = torch.exp(logp)
        return -((1 - p) ** self.gamma * logp).mean()


class DiceMultiClassLoss(_WeightedLoss):
    """
    Dice Loss pour segmentation multiclasses, avec support de weight et reduction.

    - input: logits, shape (N, C, H, W)
    - target: labels entiers {0..C-1}, shape (N, H, W)

    Arguments:
        weight (Tensor, optional): vecteur de taille C pour pondérer chaque classe.
        smooth (float): terme de lissage.
        reduction (str): 'none' | 'mean' | 'sum'.
    """

    def __init__(self, weight: torch.Tensor = None, smooth: float = 1e-6, reduction: str = 'mean'):
        super().__init__(weight=weight)
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # softmax pour avoir des probs multiclasses
        probs = F.softmax(input, dim=1)  # (N, C, H, W)
        # one-hot des cibles
        target_one_hot = torch.zeros_like(probs).scatter_(1,
                                                          target.long(),
                                                          1.0)  # (N, C, H, W)

        # calcul par classe
        dims = (0, 2, 3)  # on somme sur batch, H et W → vecteur de taille C
        intersection = (probs * target_one_hot).sum(dims)
        union = probs.sum(dims) + target_one_hot.sum(dims)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss_per_class = 1.0 - dice_score  # shape = (C,)

        # application de weight par classe si fourni
        if self.weight is not None:
            w = self.weight.to(input.device)
            loss_per_class = loss_per_class * w

        # reduction
        if self.reduction == 'mean':
            return loss_per_class.mean()
        elif self.reduction == 'sum':
            return loss_per_class.sum()
        else:  # 'none'
            return loss_per_class


class CrossEntropyDiceLoss(nn.Module):
    """
    Combinaison de CrossEntropyLoss + DiceMultiClassLoss.

    - alpha : poids de la CE (1-alpha pour le Dice).
    """

    def __init__(self,
                 alpha: float = 0.5,
                 weight: torch.Tensor = None,
                 smooth: float = 1e-6,
                 reduction: str = 'mean'):
        super().__init__()
        # CrossEntropyLoss attend des logits et cibles entières
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        # Dice multiclasses sur les mêmes poids et reduction
        self.dice = DiceMultiClassLoss(weight=weight, smooth=smooth, reduction=reduction)
        self.alpha = alpha

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(input, target)
        dice_loss = self.dice(input, target)
        return self.alpha * ce_loss + (1.0 - self.alpha) * dice_loss

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

class DeepLabV3:

    def __new__(cls, *args, **kwargs):
        model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        model.classifier = DeepLabHead(2048, num_classes=8)
        #model.classifier = nn.Conv2d(960, 8, kernel_size=1)
        model.aux_classifier = None
        return model



class SegmentedVGG16(nn.Module):
    """
    Do segmentation in label mask shape (B,C,H,W) channel first
    """
    skip_connexion: bool = True
    upsampling: bool = True
    def __init__(self, width, height, num_classes: int =8):
        super().__init__()
        self.input_width = width
        self.input_height = height
        self.num_classes = num_classes
        # Use batch normalization for vgg16
        self.vgg16 = models.vgg16_bn(pretrained=True)
        if self.skip_connexion:
            # Allow to compute an auxiliary loss to force to nn to predict better prediction sooner
            # Get direct logits, no relu and batchnorm, linear transformation
            self.aux_head = nn.Sequential(
                nn.Conv2d(64, num_classes, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # pour revenir à (H, W)
            )
            # Décomposer l'encoder
            self.enc1 = self.vgg16.features[0:6]  # conv1 (64)
            self.enc2 = self.vgg16.features[6:13]  # conv2 (128)
            self.enc3 = self.vgg16.features[13:23]  # conv3 (256)
            self.enc4 = self.vgg16.features[23:33]  # conv4 (512)
            self.enc5 = self.vgg16.features[33:43]  # conv5 (512)

            # Decoder
            # 512 + 512 allow getting global and detail information, what is it and where is it
            self.up4 = self._upsample_block(512 + 512, 256)
            self.up3 = self._upsample_block(256 + 256, 128)
            self.up2 = self._upsample_block(128 + 128, 64)
            self.up1 = self._upsample_block(64 + 64, 64)
            self.drop_up4 = nn.Dropout(0.05)
            self.drop_up3 = nn.Dropout(0.05)
            self.drop_up2 = nn.Dropout(0.05)
            self.drop_up1 = nn.Dropout(0.05)
            # Linear reduction to num_class without creation information
            self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        elif self.upsampling:
            # Decoder allow a progressive upsampling
            self.decoder = self.simple_upsampling()
        else:
            # Heavy compression (Conv → ReLU → Conv → ReLU → MaxPool)*5
            self.encoder = self.vgg16.features
            # 512 to 8 classes, no upsampling
            self.classifier = nn.Conv2d(512, self.num_classes, kernel_size=1)
    @staticmethod
    def _upsample_block(in_channels, out_channels):
        """
        Sequence of Conv -> Upsampling -> ReLU
        :return:
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # Batch normalisation to stabilize
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def simple_upsampling(self):
        return nn.Sequential(
            self._upsample_block(512, 256),
            self._upsample_block(256, 128),
            nn.Conv2d(128, self.num_classes, 1)
        )

    def skip_connexion_encoder(self, x):
        """
        Upsampling retrieving each step of the cnn
        :param x:
        :return:
        """
        x1 = self.enc1(x)  # H/2
        x2 = self.enc2(x1)  # H/4
        x3 = self.enc3(x2)  # H/8
        x4 = self.enc4(x3)  # H/16
        x5 = self.enc5(x4)  # H/32
        return x1, x2, x3, x4, x5

    def interpolation(self, x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def skip_connexion_decoder(self, x1, x2, x3, x4, x5):
        # x5 + x4 allow getting global and detail information, what is it and where is it
        d4 = self.up4(torch.cat([self.interpolation(x5, x4.shape[2:]), x4], dim=1))  # H/16
        #d4 = self.drop_up4(d4)
        d3 = self.up3(torch.cat([self.interpolation(d4, x3.shape[2:]), x3], dim=1))  # H/8
        #d3 = self.drop_up3(d3)
        d2 = self.up2(torch.cat([self.interpolation(d3, x2.shape[2:]), x2], dim=1))  # H/4
        #d2 = self.drop_up2(d2)
        d1 = self.up1(torch.cat([self.interpolation(d2, x1.shape[2:]), x1], dim=1))  # H/2
        #d1 = self.drop_up1(d1)
        aux_out = self.aux_head(d2)
        return d1, aux_out


    def forward(self, x):
        """"""
        aux_out = None
        if self.skip_connexion:
            x, aux_out = self.skip_connexion_decoder(*self.skip_connexion_encoder(x))
            x = self.final_conv(x)
            aux_out = self.interpolation(aux_out, size=(self.input_height, self.input_width))
        else:
            # Heavy compression, Loss details, output (1, 512, 7, 7)
            x = self.encoder(x)
            if self.upsampling:
                x = self.decoder(x) if self.upsampling else self.classifier(x)

        # from cats number to size of the image, Upsampling brutal
        if x.shape[2:] != (self.input_height, self.input_width):
            x = self.interpolation(x, size=(self.input_height, self.input_width))
        return x, aux_out


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