"""
Project configuration
"""
import sys
import os
import torch
import glob
import logging

DEV=os.environ.get("DEV", False)
# Configure logging to console output
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)

images = sorted(glob.glob("data/segmented/leftImg8bit/train/*/*.png"))
masks = sorted(glob.glob("data/annotate/gtFine/train/*/*_labelIds.png"))
logging.info(f"Images {len(images)} / Masks {len(masks)}")
API_ENDPOINT = os.environ.get('API_ENDPOINT', "http://127.0.0.1:8000/predict")
if torch.cuda.is_available():
    DEVICE = os.environ.get("DEVICE", "cuda")
elif torch.backends.mps.is_available():
    DEVICE = os.environ.get("DEVICE", "mps")
else:
    DEVICE = os.environ.get("DEVICE", "cpu")

WRAPPER_CLASS = os.environ.get('WRAPPER_NAME',
                               'SegmentedUnetWrapper')
WRAPPER_CONFIG = dict(
    x_data=images,
    y_data=masks,
    distributed=False,
    crop=0,
    frac=0.5,
    experiment_name="Segmentation",
    degradation=None)
