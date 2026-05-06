"""Client 02: Capture activations from multiple layers."""
import os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import torch
from nnsight.modeling.vllm import VLLM

URL = os.environ.get("NNSIGHT_SERVE_URL", "http://127.0.0.1:6677")
model = VLLM("Qwen/Qwen3-30B-A3B")

prompt = "Mixture of Experts models route tokens to"
layers = [0, 8, 16, 24, 36, 47]

t0 = time.perf_counter()
with model.trace(prompt, serve=URL):
    h0 = model.model.layers[0].output[0].save()
    h8 = model.model.layers[8].output[0].save()
    h16 = model.model.layers[16].output[0].save()
    h24 = model.model.layers[24].output[0].save()
    h36 = model.model.layers[36].output[0].save()
    h47 = model.model.layers[47].output[0].save()
elapsed = time.perf_counter() - t0

saved = {0: h0, 8: h8, 16: h16, 24: h24, 36: h36, 47: h47}

print(f"[{elapsed:.3f}s] Captured {len(layers)} layers for: '{prompt}'")
for i in layers:
    h = saved[i]
    print(f"  Layer {i:2d}: shape={list(h.shape)}, mean={h.float().mean():.4f}, std={h.float().std():.4f}")

# Verify layers are distinct
for a, b in zip(layers[:-1], layers[1:]):
    assert not torch.equal(saved[a], saved[b]), f"Layer {a} == Layer {b}"
print("All layers distinct")
