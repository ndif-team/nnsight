import os, yaml
from .schema.config import ConfigModel

PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, "config.yaml"), "r") as file:
    CONFIG = ConfigModel(**yaml.safe_load(file))
    
from importlib.metadata import PackageNotFoundError, version
  
try:
    __version__ = version("nnsight")
except PackageNotFoundError:
    __version__ = "unknown version"

from IPython import get_ipython

try:
    __IPYTHON__ = get_ipython() is not None
except NameError:
    __IPYTHON__ = False


from .intervention.envoy import Envoy
from .modeling.base import NNsight
from .modeling.language import LanguageModel
from .intervention.tracing.base import Tracer


    
def session(*args, **kwargs):
    return Tracer(*args, **kwargs)

#TODO legacy stuff like nnsight.list