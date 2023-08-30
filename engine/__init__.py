import os

import yaml

from .modeling.Config import ConfigModel
from .monkey_patching import *

PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, "config.yaml"), "r") as file:
    CONFIG = ConfigModel(**yaml.safe_load(file))

CONFIG.APP.MODEL_CACHE_PATH = os.path.join(PATH, CONFIG.APP.MODEL_CACHE_PATH)


from .Model import Model
