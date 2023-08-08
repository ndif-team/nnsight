import os
from .monkey_patching import *
import yaml

PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, "config.yaml"), "r") as file:
    CONFIG = yaml.safe_load(file)

CONFIG['APP']['MODEL_CACHE_PATH'] = os.path.join(PATH, CONFIG['APP']['MODEL_CACHE_PATH'])

from .Model import Model
from .Module import Module
