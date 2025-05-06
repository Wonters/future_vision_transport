import torch
import torch.nn as nn
from typing import List
import numpy as np
from PIL import Image
import torch.nn.functional as F
from PIL.ImageOps import grayscale
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from .utils import degrade_png_quality, CATEGORIES_MASK, group_masked



class DatasetVGG16(Dataset):
    transform_rgb = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    transform_gray = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])
    device: str = "mps"
    def __init__(self, images, masked_images, degradation: int = 5):
        self.images = images
        self.masked_images = masked_images
        self.degradation = degradation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image_mask = Image.open(self.masked_images[idx])
        if self.degradation:
            image = degrade_png_quality(image, self.degradation)
            image_mask = degrade_png_quality(image_mask, self.degradation, mode="L")
        x_vgg16 = self.transform_rgb(image)
        y_mask = transforms.ToTensor()(image_mask)
        y_mask = y_mask.to(self.device)
        x_vgg16 = x_vgg16.to(self.device)
        mask = group_masked(y_mask)
        mask = mask.reshape(y_mask.shape[1] * y_mask.shape[2], 8)
        mask = mask.to(self.device)
        return x_vgg16, mask

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_layers, kernel_size=3, padding='same', dilation=1, bn=False, pool=False, name=''):
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
    def __init__(self, img_height, img_width, nclasses, use_ctx_module=False, bn=False):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.nclasses = nclasses
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
        self.frontend_output = nn.Conv2d(nf * 64, nclasses, kernel_size=1)

        if self.use_ctx_module:
            self.ctx1 = ConvBlock(nclasses, nclasses * 2, conv_layers=2, bn=bn, name='ctx1')
            self.ctx2 = ConvBlock(nclasses * 2, nclasses * 4, conv_layers=1, dilation=2, bn=bn, name='ctx2')
            self.ctx3 = ConvBlock(nclasses * 4, nclasses * 8, conv_layers=1, dilation=4, bn=bn, name='ctx3')
            self.ctx4 = ConvBlock(nclasses * 8, nclasses * 16, conv_layers=1, dilation=8, bn=bn, name='ctx4')
            self.ctx5 = ConvBlock(nclasses * 16, nclasses * 32, conv_layers=1, dilation=16, bn=bn, name='ctx5')
            self.ctx7 = ConvBlock(nclasses * 32, nclasses * 32, conv_layers=1, bn=bn, name='ctx7')
            self.ctx_output = nn.Conv2d(nclasses * 32, nclasses, kernel_size=1)

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

        # Reshape and softmax
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), -1, self.nclasses)
        x = F.softmax(x, dim=-1)

        return x


class SegmentedVGG16(nn.Module):
    """"""
    def __init__(self,  witdh, height, num_classes: int =8):
        super().__init__()
        self.input_width = witdh
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
        # from cats number to size of the image
        x = F.interpolate(x, size=(self.input_height, self.input_width), mode='bilinear', align_corners=False)
        # Transpose from class card to pixel card
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_classes)
        x = F.softmax(x, dim=-1)
        return x

class SegmentedModelWrapper:
    """
    Model wrapper to train and predict model with pytorch
    """
    model_class = SegmentedVGG16
    dataset_class = DatasetVGG16
    epochs: int = 1
    device: str = "mps"
    batch_size: int = 1

    def __init__(self,x_data, y_data, degradation: int = 10):
        """"""
        self.model = self.model_class(witdh=2048, height=1024, num_classes=8)
        self.model.to(self.device)
        self.dataset = self.dataset_class(x_data, y_data, degradation=degradation)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2.e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.2)

    def train(self):
        """"""
        self.model.train()
        for epoch in range(self.epochs):
            for images, masks in self.dataloader:
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, masks)
                print(f"Epoch {epoch} : Loss {loss.item()}")
                break
                loss.backward()
                self.optimizer.step()
            self.scheduler.step(epoch)
        torch.save(self.model.state_dict(), 'model.pth')

    def predict(self, images: List[str]):
        """"""
        state_dict = torch.load('model.pth')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        for i in range(0, len(images), self.batch_size):
            x_vgg16 = torch.stack([DatasetVGG16.transform_rgb(Image.open(image_path))
                                       for image_path in images[i:i+self.batch_size]])
            x_vgg16 = x_vgg16.to(self.device)
            with torch.no_grad():
                output = self.model(x_vgg16)
        return output.reshape(-1,self.model.input_height, self.model.input_width, self.model.num_classes)


class SegmentedVgg16Wrapper(SegmentedModelWrapper):
    """
    Wrapper for vgg16
    """
    model_class = SegmentedVGG16
    dataset_class = DatasetVGG16

class SegmentedDilatednetWrapper(SegmentedModelWrapper):
    """
    Wrapper for unet
    """
    model_class = DilatedNet
    dataset_class = DatasetVGG16