"""Classes the serialization tests send through a payload.

A test that runs a deserialized tracer in-process registers the block's source in
``linecache`` under the test file's own path, as a server does for a file it does
not have. The source of anything else defined in that file then reads back wrong,
and a class defined in a script travels by source, so these live in their own file.
"""

from nnsight.intervention.envoy import Envoy
from nnsight.intervention.eproperty import eproperty
from nnsight.modeling.transformers import TransformersModel


class Doubled(Envoy):
    """An envoy class a caller's ``envoys=`` map chooses for the MLPs."""

    @eproperty(key="output")
    def doubled(self, value):
        return value * 2


class WrappedModel(TransformersModel):
    """A model wrapper a receiver's stock tree does not build."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stack = self.transformer.h

    def last_block(self):
        return self.transformer.h[-1]

    @eproperty(key="output")
    def doubled_logits(self, value):
        return value.logits * 2


def script_class():
    """A class as a notebook defines one: dynamic, so it travels by source."""

    class Local(Envoy):
        @eproperty(key="output")
        def tripled(self, value):
            return value * 3

    return Local
