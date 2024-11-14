
"""
The `intervention` module extends the `tracing` module to add PyTorch specific interventions to a given computation graph.
It defines its own: protocols, contexts, backends and graph primitives to achieve this.
"""
from .base import NNsight
from .envoy import Envoy