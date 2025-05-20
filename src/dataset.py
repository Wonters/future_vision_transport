import random
import numpy as np
import torch
from PIL import Image
import logging
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.v2.functional import crop_image
from src.utils import degrade_png_quality, group_mask
from src.config import DEVICE

logger = logging.getLogger(__name__)


class DatasetVGG16(Dataset):
    """
    Dataset to prepare images for segmentation training
    You can decrease the quality using degradation input
    """
    transform_rgb = transforms.Compose([
        #transforms.RandomRotation(10),
        #transforms.RandomCrop((224, 224)),
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

    device: str = DEVICE

    def __init__(self, images, masked_images, degradation: int = 5, crop=1024):
        self.images = images
        self.masked_images = masked_images
        self.degradation = degradation
        self.crop = crop
        self.random_rotation = random.randrange(-10, 10)
        self.random_crop = random.randrange(0, 100), random.randrange(0, 512)

    def __len__(self):
        return len(self.images)

    def square_crop(self, image, top=0, left=0, size=1024) -> torch.Tensor:
        """
        Return a squared image
        :param image: image to crop
        :param top:
        :param left:
        :param size:
        """
        t = torch.from_numpy(np.array(image))
        if len(t.shape) == 3:
            return crop_image(t.permute(2, 0, 1), top=top, left=left, height=size, width=size).permute(1, 2, 0)
        else:
            return crop_image(t.unsqueeze(0), top=top, left=left, height=size, width=size)[0, ...]

    def __getitem__(self, idx):
        """
        Return the image without mask as x and the mask as y
        :param idx:
        :return:
        """
        image = Image.open(self.images[idx])
        image_mask = Image.open(self.masked_images[idx])
        if self.degradation:
            image = degrade_png_quality(image, self.degradation)
            image_mask = degrade_png_quality(image_mask, self.degradation, mode="L")
        if self.random_crop:
            top, left = self.random_crop
            # todo: min(image.size)-top -> change it for image higher than width
            image = Image.fromarray(self.square_crop(image, top=top, left=left,
                                                     size=min(image.size) - top).numpy())
            image_mask = Image.fromarray(self.square_crop(image_mask, top=top, left=left,
                                                          size=min(image_mask.size) - top).numpy())
        elif self.crop:
            image = self.square_crop(image)
        if self.random_rotation:
            image = transforms.functional.rotate(image,
                                                 angle=self.random_rotation,
                                                 interpolation=transforms.InterpolationMode.NEAREST,
                                                 expand=False,
                                                 fill=(0,0,0))
            image_mask = transforms.functional.rotate(image_mask,
                                                      angle=self.random_rotation,
                                                      interpolation=transforms.InterpolationMode.NEAREST,
                                                      expand=False,
                                                      fill=(0,))
        if isinstance(image, torch.Tensor):
            image = Image.fromarray(image.numpy())
        x_vgg16 = self.transform_rgb(image)
        y_mask = torch.from_numpy(np.array(image_mask))
        y_mask = y_mask.unsqueeze(0)
        mask = group_mask(y_mask)
        mask = mask[0, ...]
        mask = mask.permute(2, 0, 1)
        mask = mask.to(self.device)
        x_vgg16 = x_vgg16.to(self.device)
        return x_vgg16, mask, idx


class DatasetDeepLabV3(DatasetVGG16):
    transform_rgb = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    transform_gray = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])
