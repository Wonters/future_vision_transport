"""
Project configuration
"""
import os
import glob
from src.utils import load_module
import logging

# Configure logging to console output
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)

images = glob.glob("zurich/annotat/*.png")
masks = glob.glob("zurich/segmented/*_labelIds.png")

API_ENDPOINT = os.environ.get('API_ENDPOINT', "http://127.0.0.1:8001/predict")
DEVICE = os.environ.get("DEVICE", "mps")

WRAPPER_CLASS = os.environ.get('WRAPPER_NAME',
                               'SegmentedUnetWrapper')
WRAPPER_CONFIG = dict(
    x_data=images,
    y_data=masks,
    distributed=False,
    frac=0.3,
    degradation=5)
