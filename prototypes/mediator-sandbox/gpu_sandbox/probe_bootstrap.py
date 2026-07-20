#!/usr/bin/env python3
"""SP1 bootstrap probe — can a per-mediator intervention be serialized, then
deserialized against DUMMY modules + a stub interleaver, and run far enough to
call request() with the correct requester string?

This de-risks the worker bootstrap (the hardest unknown) WITHOUT the channel,
forward, or saves. If this prints the expected requester, the rest is plumbing
we already have (CudaIpcChannel + Shim A/B).

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
    /disk/u/zikai/anaconda3/envs/hf-serve/bin/python \
    prototypes/mediator-sandbox/gpu_sandbox/probe_bootstrap.py
"""
import sys
import types

import torch
import torch.nn as nn

from nnsight import LanguageModel
from nnsight.intervention import serialization
from nnsight.intervention.interleaver import Mediator
from nnsight.intervention.tracing.globals import _ensure_mounted


def main():
    model = LanguageModel("gpt2", device_map="cpu", dispatch=True)

    # 1. Capture a real mediator's serialized bytes during a normal trace.
    captured = {}
    orig_start = Mediator.start

    def patched_start(self, interleaver):
        if "bytes" not in captured:
            try:
                # The tracer attaches source during ITS __getstate__ (tracer.py:677);
                # per-mediator serialization must do the same first.
                self.intervention.__source__ = "".join(self.info.source)
                captured["bytes"] = serialization.dumps(self)
                captured["err"] = None
            except Exception as e:  # noqa: BLE001
                captured["err"] = f"{type(e).__name__}: {e}"
        return orig_start(self, interleaver)

    Mediator.start = patched_start
    try:
        with model.trace("The Eiffel Tower is in"):
            model.transformer.h[6].output.save()
    finally:
        Mediator.start = orig_start

    if captured.get("err"):
        print(f"[serialize] FAILED: {captured['err']}")
        sys.exit(1)
    print(f"[serialize] mediator -> {len(captured['bytes'])} bytes OK")

    # 2. Build a WORKER-style persistent map: dummy modules for every Module:<path>,
    #    a stub interleaver, real tokenizer/processor pass-through.
    real_map = model._remoteable_persistent_objects()

    calls = []

    class StubBatcher:
        current_provider = None
        current_value = None

    class StubInterleaver:
        interleaving = True
        batcher = StubBatcher()
        current = None

        def iterate_requester(self, requester):
            med = self.current
            iteration = med.iteration if med.iteration is not None else med.iteration_tracker[requester]
            return f"{requester}.i{iteration}"

    stub_interleaver = StubInterleaver()

    pmap = {}
    for k, v in real_map.items():
        if k.startswith("Module:"):
            dummy = nn.Module()
            dummy.__path__ = k[len("Module:") :]
            pmap[k] = dummy
        elif k == "Interleaver":
            pmap[k] = stub_interleaver
        else:
            pmap[k] = v  # Tokenizer / Processor — keep real (lightweight)

    # 3. Deserialize the mediator against dummies.
    try:
        med = serialization.loads(captured["bytes"], pmap)
    except Exception as e:  # noqa: BLE001
        print(f"[deserialize] FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
    print(f"[deserialize] mediator rebuilt against dummy modules OK (name={med.name})")

    # 4. Wire it into the stub interleaver and run the intervention with a
    #    recording request() — assert it asks for the right provider.
    _ensure_mounted()
    med.idx = 0
    med.interleaver = stub_interleaver
    stub_interleaver.current = med

    def recording_request(requester):
        calls.append(requester)
        return torch.zeros(1, 16, 768)  # gpt2-ish block output stand-in

    med.request = recording_request
    med.swap = lambda requester, value: calls.append(("swap", requester))
    med.end = lambda: None
    med.push = lambda: None
    med.pull = lambda: None
    med.cross_invoker = False

    try:
        med.intervention(med, med.info)
    except Exception as e:  # noqa: BLE001
        print(f"[run] intervention raised: {type(e).__name__}: {e}")
        print(f"[run] calls so far: {calls}")
        sys.exit(1)

    expected = "model.transformer.h.6.output.i0"  # real envoy root prefixes "model."
    ok = expected in calls
    print(f"[run] intervention requested: {calls}")
    print("=" * 72)
    print(f"BOOTSTRAP PROBE: {'PASS' if ok else 'FAIL'} (expected {expected!r})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
