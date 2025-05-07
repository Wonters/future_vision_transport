import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils import degrade_png_quality, group_masked


class DatasetVGG16(Dataset):
    """
    Dataset to prepare images for segmentation training
    You can decrease the quality using degradation input
    """
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
        x_vgg16 = self.transform_rgb(image)
        y_mask = torch.from_numpy(np.array(image_mask))
        y_mask = y_mask.unsqueeze(0)
        mask = group_masked(y_mask)
        mask = mask.permute(2,0,1)
        #mask = mask.reshape(y_mask.shape[1] * y_mask.shape[2], 8)
        mask = mask.to(self.device)
        x_vgg16 = x_vgg16.to(self.device)
        return x_vgg16, mask
