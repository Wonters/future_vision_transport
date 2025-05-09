from src.config import WRAPPER_CONFIG, WRAPPER_CLASS
from src.utils import load_module
wrapper = load_module('src.wrapper',WRAPPER_CLASS)(**WRAPPER_CONFIG)

wrapper.train()