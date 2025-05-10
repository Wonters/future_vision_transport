"""
Project configuration
"""
import os
import glob
import logging
import logging

# Configure logging to console output
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)

images = glob.glob("data/segmented/leftImg8bit/train/*/*.png")
masks = glob.glob("data/annotate/gtFine/train/*/*_labelIds.png")
logging.info(f"Images {len(images)} / Masks {len(masks)}")
API_ENDPOINT = os.environ.get('API_ENDPOINT', "http://127.0.0.1:8001/predict")
DEVICE = os.environ.get("DEVICE", "mps")

WRAPPER_CLASS = os.environ.get('WRAPPER_NAME',
                               'SegmentedUnetWrapper')
WRAPPER_CONFIG = dict(
    x_data=images,
    y_data=masks,
    distributed=False,
    frac=0.5,
    degradation=5)
