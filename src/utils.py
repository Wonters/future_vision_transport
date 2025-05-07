from PIL import Image
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

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

LAYER_COLORS = {
'void': (255,100,255),
 'flat': (255,255,255),
 'construction': (0,255,255),
 'object': (255,0,255),
 'nature': (0,255,0),
 'sky': (0,0,255),
 'human': (255,0,0),
 'vehicle': (255,0,255)
}

def degrade_png_quality(png_image: Image.Image, quality=10, mode="RGB") -> Image.Image:
    """
    Decrease the quality of an image
    :param png_image:
    :param quality:
    :param mode:
    :return:
    """
    # Convertir en RGB si nécessaire (JPEG ne supporte pas la transparence)
    degraded_image = np.array(png_image)[::quality, ::quality]
    img_degraded = Image.fromarray(degraded_image, mode=mode).resize(png_image.size, Image.BILINEAR)
    return img_degraded

def group_masked(y_mask, with_cat_number:bool=False):
    """
    Return a grouped categories mask
    :param y_mask:
    :param with_cat_number: fill the mask with the value of the category instead of 1
    :return:
    """
    mask = torch.zeros((*y_mask.shape[1::], 8))
    for i in range(-1, 34):
        for j, (cat_name, cat_values) in enumerate(CATEGORIES_MASK.items()):
            if i in cat_values:
                sub = torch.logical_or(mask[:, :, j], (y_mask == i))
                if with_cat_number:
                    sub = sub.to(torch.int8)
                    sub[sub==1] = j
                mask[:, :, j] = sub
    return mask

def iou_score(pred, target, eps=1e-6):
    """
    Compute IOU score
    :param pred:
    :param target:
    :param eps:
    :return:
    """
    # pred et target : booléens ou 0/1, même shape
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    return (intersection + eps) / (union + eps)
