"""Constructor routing through the Loadable/Meta mixins.

Three behaviors, each pinned against a stub model class with nonstandard
module names: a ready module builds the tree directly (never enters `_load`),
Envoy's keyword arguments never reach the load path, and both hold across a
lazy construction's dispatch replay.
"""

import torch
import torch.nn as nn

from nnsight.intervention.interleaver import Interleaver
from nnsight.modeling.mixins.loadable import Loadable
from nnsight.modeling.mixins.meta import Meta


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_blocks = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
        self.output_projection = nn.Linear(3, 5)

    def forward(self, x):
        return self.output_projection(self.decoder_blocks(x))


class SpyLoadable(Loadable):
    """Builds TinyNet from a size argument; records what `_load` received."""

    def _load(self, hidden, **kwargs):
        self.load_seen = (hidden, dict(kwargs))
        return TinyNet()


class SpyMeta(Meta):
    def _load_meta(self, hidden, **kwargs):
        self.meta_seen = dict(kwargs)
        return TinyNet()

    def _load(self, hidden, **kwargs):
        self.load_seen = dict(kwargs)
        return TinyNet()


def test_ready_module_builds_the_tree_directly():
    module = TinyNet()
    m = SpyLoadable(module)
    assert m._module is module
    assert not hasattr(m, "load_seen")

    with m.trace(torch.randn(1, 3)):
        out = m.output_projection.output.save()
    assert out.shape == (1, 5)


def test_envoy_kwargs_never_reach_the_load_path():
    custom = Interleaver()
    m = SpyLoadable(3, interleaver=custom, flavor="q8")
    assert m.load_seen == (3, {"flavor": "q8"})
    assert m.interleaver is custom


def test_lazy_construction_and_dispatch_keep_the_buckets():
    custom = Interleaver()
    m = SpyMeta(3, interleaver=custom, flavor="q8")
    assert m.interleaver is custom
    assert m.meta_seen == {"flavor": "q8"}

    m.dispatch()
    assert m.load_seen == {"flavor": "q8"}
    assert m.interleaver is custom
