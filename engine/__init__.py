import os

import yaml

PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, "config.yaml"), "r") as file:
    CONFIG = yaml.safe_load(file)

from .Model import Model
from .Module import Module
