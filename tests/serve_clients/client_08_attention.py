"""Client 08: Capture attention layer outputs across multiple layers."""
import os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import torch
from nnsight.modeling.vllm import VLLM

URL = os.environ.get("NNSIGHT_SERVE_URL", "http://127.0.0.1:6677")
model = VLLM("Qwen/Qwen3-30B-A3B")

prompt = "Attention mechanisms in transformers allow the model to"

t0 = time.perf_counter()
with model.trace(prompt, serve=URL):
    a0 = model.model.layers[0].self_attn.output[0].save()
    a12 = model.model.layers[12].self_attn.output[0].save()
    a24 = model.model.layers[24].self_attn.output[0].save()
    a36 = model.model.layers[36].self_attn.output[0].save()
    a47 = model.model.layers[47].self_attn.output[0].save()
elapsed = time.perf_counter() - t0

attn_outs = {0: a0, 12: a12, 24: a24, 36: a36, 47: a47}
layers = [0, 12, 24, 36, 47]

print(f"[{elapsed:.3f}s] Attention outputs for: '{prompt}'")
for i in layers:
    a = attn_outs[i]
    print(f"  Layer {i:2d} attn: shape={list(a.shape)}, "
          f"norm={a.float().norm():.2f}, "
          f"max={a.float().abs().max():.4f}")

for j in range(len(layers) - 1):
    a, b = layers[j], layers[j + 1]
    assert not torch.equal(attn_outs[a], attn_outs[b]), f"Layer {a} == Layer {b}"
print("All attention outputs distinct")
