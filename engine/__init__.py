import os

import yaml

PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, "config.yaml"), "r") as file:
    CONFIG = yaml.safe_load(file)

from transformers.utils import hub

CONFIG['APP']['MODEL_CACHE_PATH'] = os.path.join(PATH, CONFIG['APP']['MODEL_CACHE_PATH'])

hub.PYTORCH_TRANSFORMERS_CACHE = CONFIG['APP']['MODEL_CACHE_PATH']
hub.TRANSFORMERS_CACHE = CONFIG['APP']['MODEL_CACHE_PATH']
hub.HUGGINGFACE_HUB_CACHE = CONFIG['APP']['MODEL_CACHE_PATH']

from .Model import Model
from .Module import Module
