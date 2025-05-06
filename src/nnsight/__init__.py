from .intervention.envoy import Envoy
from .modeling.base import NNsight
from .modeling.language import LanguageModel
from .intervention.tracing.base import Tracer

def session(*args, **kwargs):
    return Tracer(*args, **kwargs)

#TODO legacy stuff like nnsight.listp