import os

import yaml

from .patching import *
from .pydantics.Config import ConfigModel

PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, "config.yaml"), "r") as file:
    CONFIG = ConfigModel(**yaml.safe_load(file))

from .logger import logger
from .models.NNsightModel import NNsight
from .models.LanguageModel import LanguageModel

from .module import Module
from .patching import Patch, Patcher
from .tracing.Proxy import proxy_wrapper

logger.disabled = not CONFIG.APP.LOGGING

# Below do default patching:
DEFAULT_PATCHER = Patcher()

from inspect import getmembers, isfunction

import einops

for key, value in getmembers(einops.einops, isfunction):
    DEFAULT_PATCHER.add(Patch(einops.einops, proxy_wrapper(value), key))


from torch._subclasses.fake_tensor import FakeTensor


def _bool(self):
    return True


DEFAULT_PATCHER.add(Patch(FakeTensor, _bool, "__bool__"))

DEFAULT_PATCHER.__enter__()
