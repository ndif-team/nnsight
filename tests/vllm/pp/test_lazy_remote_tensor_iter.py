"""Unit tests for LazyRemoteTensor iteration / length protocol.

Regression for the cross-stage replacement-write hang: on the non-owning PP
rank a layer output is a LazyRemoteTensor. A write like

    layer.output = (out[0] + delta,) + tuple(out[1:])

evaluates ``tuple(out[1:])`` on that rank. Before the fix, LazyRemoteTensor
defined ``__getitem__`` (returning a fresh lazy, never raising IndexError) but
no ``__iter__``, so Python's sequence-protocol fallback made ``tuple(lazy)`` /
``list(lazy)`` / ``for`` / unpacking spin forever — while the owning rank,
iterating the real value, terminated. That divergence hung the trace until the
driver timed out (~550s). The fix makes iteration a materializing operation.

These tests exercise that contract directly — no GPU, no PP.

Run: python -m pytest tests/test_lazy_remote_tensor_iter.py
"""

import torch

from nnsight.modeling.vllm.lazy_remote_tensor import LazyRemoteTensor


def _lazy_for(real, key="m.output.i0"):
    """A LazyRemoteTensor whose pull returns ``real`` (a tuple or tensor)."""
    lz = LazyRemoteTensor(source_rank=0, provider_string=key, dtype=torch.float32)
    lz._pull_fn = lambda _src, _prov: real
    return lz


def test_tuple_of_slice_terminates_and_matches_real():
    # The exact failing idiom: out is a (hidden, residual) layer output;
    # tuple(out[1:]) must terminate and equal the real tail.
    hidden = torch.randn(3, 4)
    residual = torch.randn(3, 4)
    out = _lazy_for((hidden, residual))

    rest = tuple(out[1:])
    assert len(rest) == 1
    assert torch.equal(rest[0], residual)


def test_iter_yields_real_elements():
    hidden = torch.randn(2, 5)
    residual = torch.randn(2, 5)
    out = _lazy_for((hidden, residual))

    elems = [t for t in out]
    assert len(elems) == 2
    assert torch.equal(elems[0], hidden)
    assert torch.equal(elems[1], residual)


def test_unpacking_works():
    hidden = torch.randn(1, 3)
    residual = torch.randn(1, 3)
    out = _lazy_for((hidden, residual))

    a, b = out
    assert torch.equal(a, hidden)
    assert torch.equal(b, residual)


def test_list_of_lazy_terminates():
    real = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))
    out = _lazy_for(real)
    assert len(list(out)) == 3


def test_len_reflects_real_value():
    out = _lazy_for((torch.randn(2, 2), torch.randn(2, 2)))
    assert len(out) == 2


def test_iter_over_bare_tensor_matches_torch_rows():
    # For tensor-output models (e.g. gpt2) the layer output is a bare tensor;
    # iterating must mirror torch's row iteration so both ranks agree.
    real = torch.randn(4, 6)
    out = _lazy_for(real)

    rows = list(out)
    assert len(rows) == 4
    assert torch.equal(torch.stack(rows), real)


def test_iter_reuses_cached_materialization():
    # _materialize caches in _real; iterating twice must not re-pull.
    calls = {"n": 0}
    real = (torch.randn(2, 2), torch.randn(2, 2))
    lz = LazyRemoteTensor(source_rank=0, provider_string="k", dtype=torch.float32)

    def _pull(_src, _prov):
        calls["n"] += 1
        return real

    lz._pull_fn = _pull
    _ = list(lz)
    _ = list(lz)
    assert calls["n"] == 1
