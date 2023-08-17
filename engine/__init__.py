import logging
import os

import yaml

from .modeling import ConfigModel
from .monkey_patching import *

PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, "config.yaml"), "r") as file:
    CONFIG = ConfigModel(**yaml.safe_load(file))

CONFIG.APP.MODEL_CACHE_PATH = os.path.join(PATH, CONFIG.APP.MODEL_CACHE_PATH)

logging_handler = logging.FileHandler(os.path.join(PATH, f"engine.log"), "a")
logging_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s"
    )
)
logging_handler.setLevel(logging.DEBUG)
logger = logging.getLogger("engine")
logger.addHandler(logging_handler)
logger.setLevel(logging.DEBUG)

from .Model import Model