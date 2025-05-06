from PIL import Image
import numpy as np
import torch

## Based on cityscapes dataset segmentation
CATEGORIES_POLYGON = {
    'flat': {'road', 'sidewalk', 'parking', 'rail track'},
    'human': {'person', 'rider'},
    'vehicle': {'car', 'truck', 'bus', 'on rails', 'motorcycle', 'bicycle', 'caravan', 'trailer'},
    'construction': {'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel'},
    'object': {'pole', 'pole group', 'traffic sign', 'traffic light'},
    'nature': {'vegetation', 'terrain'},
    'sky': {'sky'},
    'void': {'ground', 'dynamic', 'static'}
}
CATEGORIES_MASK = {'void': [0, 1, 2, 3, 4, 5, 6],
 'flat': [7, 8, 9, 10],
 'construction': [11, 12, 13, 14, 15, 16],
 'object': [17, 18, 19, 20],
 'nature': [21, 22],
 'sky': [23],
 'human': [24, 25],
 'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]}

def degrade_png_quality(png_image: Image.Image, quality=10, mode="RGB") -> Image.Image:
    # Convertir en RGB si nécessaire (JPEG ne supporte pas la transparence)
    degraded_image = np.array(png_image)[::quality, ::quality]
    img_degraded = Image.fromarray(degraded_image, mode=mode).resize(png_image.size, Image.BILINEAR)
    return img_degraded

def group_masked(y_mask):
    mask = torch.zeros((y_mask.shape[1], y_mask.shape[2], 8))
    for i in range(-1, 34):
        for j, (cat_name, cat_values) in enumerate(CATEGORIES_MASK.items()):
            if i in cat_values:
                mask[:, :, j] = torch.logical_or(mask[:, :, j], y_mask == i)