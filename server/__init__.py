import os

import yaml

PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, "config.yaml"), "r") as file:
    CONFIG = yaml.safe_load(file)

from .ResponseDict import ResponseDict
from .processors.SignalProcessor import SignalProcessor
from .processors.InferenceProcessor import InferenceProcessor
from .processors.RequestProcessor import RequestProcessor
