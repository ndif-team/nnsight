"""Isolate the interleaver's per-visit Python cost.

The vLLM capture benchmark showed nnsight paying ~2.5x bare vLLM at TP=1 and
~4.7x at TP=2 for a job vllm-lens does at 1.7x/2.2x.  The suspected cause is
structural rather than GPU-side: nnsight installs hooks on *every* module and
`Interleaver.handle` fans each visit out to *every* mediator, so the work per
forward scales with modules x invokes.

This reproduces that shape on CPU with tensors small enough that the Python path
dominates, so an optimization can be measured in seconds instead of GPU-minutes.

    python bench_micro.py [--layers 32] [--invokes 64] [--repeats 5]
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn as nn

import nnsight
from nnsight import NNsight

HIDDEN = 8


class Block(nn.Module):
    """A decoder-ish block: enough submodules to match a real model's count."""

    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(HIDDEN, HIDDEN)
        self.k = nn.Linear(HIDDEN, HIDDEN)
        self.v = nn.Linear(HIDDEN, HIDDEN)
        self.o = nn.Linear(HIDDEN, HIDDEN)
        self.up = nn.Linear(HIDDEN, HIDDEN)
        self.down = nn.Linear(HIDDEN, HIDDEN)
        self.n1 = nn.LayerNorm(HIDDEN)
        self.n2 = nn.LayerNorm(HIDDEN)

    def forward(self, x):
        h = self.n1(x)
        h = self.o(self.q(h) + self.k(h) + self.v(h))
        x = x + h
        return x + self.down(self.up(self.n2(x)))


class Net(nn.Module):
    def __init__(self, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Block() for _ in range(layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BatchedNet(NNsight):
    """Rows-are-invokes batching, so one forward serves many mediators.

    That is the shape the vLLM runtime produces (one flat batch, one mediator
    per request) and the shape that makes `Interleaver.handle` fan out.
    """

    def _batch_size(self, *inputs, **kwargs) -> int:
        return inputs[0].shape[0] if inputs else 0

    def _batch(self, invokes, fn):
        rows = [inputs[0] for inputs, _ in invokes]
        return (torch.cat(rows, dim=0),), {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--invokes", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--target", type=int, default=16)
    args = ap.parse_args()

    torch.set_num_threads(1)
    net = Net(args.layers)
    model = BatchedNet(net)
    n_modules = sum(1 for _ in net.modules())

    x = torch.randn(1, HIDDEN)
    inputs = [x.clone() for _ in range(args.invokes)]
    target = model.layers[args.target]

    # Baseline: the *same single batched forward* the trace runs, with no trace
    # around it, so the difference is exactly what interleaving costs.  (Running
    # 64 separate batch-1 forwards here would compare different amounts of work.)
    batched = torch.cat(inputs, dim=0)

    def bare():
        net(batched)

    def traced():
        with model.trace() as tracer:
            acts = nnsight.save([None] * len(inputs))
            for i, inp in enumerate(inputs):
                with tracer.invoke(inp):
                    acts[i] = target.output
        return acts

    traced()  # warm

    def timeit(fn):
        ts = []
        for _ in range(args.repeats):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        return statistics.median(ts)

    b = timeit(bare)
    t = timeit(traced)
    got = traced()
    assert all(a is not None for a in got), "capture did not fill every slot"

    print(f"modules={n_modules}  layers={args.layers}  invokes={args.invokes}")
    print(f"bare forwards   {b*1000:8.1f} ms")
    print(f"traced          {t*1000:8.1f} ms   ({t/b:.1f}x bare)")
    print(f"interleave cost {(t-b)*1000:8.1f} ms")
    # Visits are per (module, side); the mediator loop runs once per visit per
    # mediator, which is the quantity an optimization has to bring down.
    visits = n_modules * 2 * args.invokes
    print(f"handle() calls  {visits:8d}   "
          f"-> {(t-b)/visits*1e6:.2f} us per mediator-visit")


if __name__ == "__main__":
    main()
