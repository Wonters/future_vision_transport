from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
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
'void': (0,0,0), # black
 'flat': (255,255,255), # white
 'construction': (0,255,255), # cyan
 'object': (255,255,0), # yellow
 'nature': (0,255,0), # green
 'sky': (0,0,255), # blue
 'human': (255,0,0), # red
 'vehicle': (255,0,255) # pink
}
def center_crop(img: Image.Image) -> Image.Image:
    """Crop une image au centre et la resize à (H, W)."""
    width, height = img.size
    # Calcul du crop central
    left = (width - min(width, height)) // 2
    top = (height - min(width, height)) // 2
    right = left + min(width, height)
    bottom = top + min(width, height)
    img_cropped = img.crop((left, top, right, bottom))
    return img_cropped

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

def group_mask(y_mask, with_cat_number:bool=False):
    """
    Return a grouped categories mask
    :param y_mask: (B,H,W)
    :param with_cat_number: fill the mask with the value of the category instead of 1
    :return:
    """
    mask = torch.zeros((*y_mask.shape, 8))
    for i in range(-1, 34):
        for j, (cat_name, cat_values) in enumerate(CATEGORIES_MASK.items()):
            if i in cat_values:
                sub = torch.logical_or(mask[..., j], (y_mask == i))
                if with_cat_number:
                    sub = sub.to(torch.int8)
                    sub[sub==1] = j+1 # avoid 0
                mask[..., j] = sub
    return mask


def group_mask_v2(y_mask, with_cat_number:bool=False):
    """
    Faster, group categories from a mask
    :param y_mask:
    :param with_cat_number:
    :return:
    """
    # Lookup table
    # We guess labels go from 0 to 255
    lut = torch.full((256,), -1, dtype=torch.int32, device=y_mask.device)

    for j, (_, values) in enumerate(CATEGORIES_MASK.items()):
        lut[torch.tensor(values)] = j

    # 2. Appliquer le LUT sur y_mask
    flat_y_mask = y_mask.view(-1)  # 1D
    # Long to insure flat_y_mask can be consider as index
    mapped_flat = lut[flat_y_mask.long()]  # 1D
    mapped = mapped_flat.view(y_mask.shape)

    num_classes = len(CATEGORIES_MASK)
    mask = torch.zeros((*y_mask.shape, num_classes), dtype=torch.bool, device=y_mask.device)

    for j in range(num_classes):
        mask[..., j] = (mapped == j)

    if with_cat_number:
        indexes = torch.arange(1, mask.shape[-1]+1).view(1, 1, 1,mask.shape[-1])
        mask = mask*indexes
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

def compute_iou(logits, target_masks, mean=True, softmax=True):
    """"""
    outputs = F.softmax(logits, dim=1) if softmax else logits
    outputs = outputs.cpu().detach()
    target_masks = target_masks.cpu().detach()
    outputs = outputs.permute(0, 2, 3, 1)
    target_masks = target_masks.permute(0, 2, 3, 1)
    argmax_indices_pred = torch.argmax(outputs, dim=-1)
    output_masks = torch.zeros_like(outputs)
    output_masks.scatter_(dim=3,
                         index=argmax_indices_pred.unsqueeze(-1),
                         src=torch.ones_like(argmax_indices_pred,
                                             dtype=output_masks.dtype).unsqueeze(-1))
    output_masks = output_masks.to(torch.uint8)
    target_masks =target_masks.to(torch.uint8)
    if not mean:
        argmax_indices_masks = torch.argmax(target_masks, dim=-1)
        iou = {}
        for index, name in enumerate(CATEGORIES_MASK):
            sub_pred = argmax_indices_pred == index
            sub_target = argmax_indices_masks == index
            iou[name] = iou_score(sub_pred, sub_target)
        return iou
    return iou_score(output_masks, target_masks)

def dice(pred, target, eps=1e-6):
    """
    Compute dice score
    :param pred: probabilities
    :param target:
    :param eps:
    :return:
    """
    # pred et target : booléens ou 0/1, même shape
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    return (2 * intersection + eps) / (union + eps)


def load_module(module_name:str = "src.deeplearning", class_name:str= "Unet"):
    """
    Dynamic load model
    :param module_name:
    :param class_name:
    :return:
    """
    module = __import__(module_name, fromlist=[''])
    return getattr(module, class_name)