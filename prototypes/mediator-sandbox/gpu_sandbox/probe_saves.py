#!/usr/bin/env python3
"""SP1 saves probe — after a deserialized intervention runs to end()/push() in a
worker-like context, WHERE do the .save()'d values land, and can we filter them by
Globals.saves to ship back? (Shim B feasibility.)

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/hf-serve/bin/python \
    prototypes/mediator-sandbox/gpu_sandbox/probe_saves.py
"""
import sys

import torch
import torch.nn as nn

from nnsight import LanguageModel
from nnsight.intervention import serialization
from nnsight.intervention.interleaver import Mediator
from nnsight.intervention.tracing.globals import Globals, _ensure_mounted


def main():
    model = LanguageModel("gpt2", device_map="cpu", dispatch=True)

    captured = {}
    orig_start = Mediator.start

    def patched_start(self, interleaver):
        if "bytes" not in captured:
            self.intervention.__source__ = "".join(self.info.source)
            captured["bytes"] = serialization.dumps(self)
        return orig_start(self, interleaver)

    Mediator.start = patched_start
    try:
        with model.trace("The Eiffel Tower is in"):
            saved_marker = model.transformer.h[6].output.save()
    finally:
        Mediator.start = orig_start

    real_map = model._remoteable_persistent_objects()

    class StubBatcher:
        current_provider = None
        current_value = None

    class StubInterleaver:
        interleaving = True
        batcher = StubBatcher()
        current = None

        def iterate_requester(self, requester):
            med = self.current
            it = med.iteration if med.iteration is not None else med.iteration_tracker[requester]
            return f"{requester}.i{it}"

    stub_interleaver = StubInterleaver()
    pmap = {}
    for k, v in real_map.items():
        if k.startswith("Module:"):
            d = nn.Module(); d.__path__ = k[len("Module:"):]; pmap[k] = d
        elif k == "Interleaver":
            pmap[k] = stub_interleaver
        else:
            pmap[k] = v

    med = serialization.loads(captured["bytes"], pmap)
    _ensure_mounted()
    Globals.saves.clear()
    med.idx = 0
    med.interleaver = stub_interleaver
    stub_interleaver.current = med
    med.cross_invoker = False

    sentinel = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)  # the "activation"
    med.request = lambda requester: sentinel

    # Let the REAL end()/push() run; stub only the channel.
    class FakeChannel:
        def put_event(self, item): self.last = item
    med.channel = FakeChannel()

    med.intervention(med, med.info)

    # Shim B candidate: filter the worker's frame locals by Globals.saves.
    frame = med.info.frame
    flocals = getattr(frame, "f_locals", {})
    saved = {k: v for k, v in flocals.items() if id(v) in Globals.saves}

    print(f"[frame]   info.frame type={type(frame).__name__}, #locals={len(flocals)}")
    print(f"[saves]   Globals.saves size={len(Globals.saves)}; filtered saved keys={list(saved.keys())}")
    hit = [k for k, v in saved.items() if torch.is_tensor(v) and torch.equal(v, sentinel)]
    ok = len(hit) >= 1
    print(f"[match]   a saved local equals the sentinel activation: {ok} (keys={hit})")
    print("=" * 72)
    print(f"SAVES PROBE: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
