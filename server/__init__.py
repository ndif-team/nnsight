import os

import yaml

from .modeling import ConfigModel

# Get a parse config in same directory
PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, "config.yaml"), "r") as file:
    CONFIG = ConfigModel(**yaml.safe_load(file))

# Create log directory if needed
os.makedirs(CONFIG.LOG_PATH, exist_ok=True)

