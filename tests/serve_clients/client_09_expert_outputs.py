"""Client 09: Capture MoE gate/router outputs to see expert routing."""
import os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import torch
from nnsight.modeling.vllm import VLLM

URL = os.environ.get("NNSIGHT_SERVE_URL", "http://127.0.0.1:6677")
model = VLLM("Qwen/Qwen3-30B-A3B")

prompt = "The mixture of experts architecture selects"

t0 = time.perf_counter()
with model.trace(prompt, serve=URL):
    m0 = model.model.layers[0].mlp.output.save()
    m12 = model.model.layers[12].mlp.output.save()
    m24 = model.model.layers[24].mlp.output.save()
    m36 = model.model.layers[36].mlp.output.save()
    m47 = model.model.layers[47].mlp.output.save()
elapsed = time.perf_counter() - t0

mlp_outs = {0: m0, 12: m12, 24: m24, 36: m36, 47: m47}
layers = [0, 12, 24, 36, 47]

print(f"[{elapsed:.3f}s] MoE MLP outputs for: '{prompt}'")
for i in layers:
    m = mlp_outs[i]
    if isinstance(m, torch.Tensor):
        print(f"  Layer {i:2d} MoE MLP: shape={list(m.shape)}, "
              f"norm={m.float().norm():.2f}, "
              f"sparsity={(m == 0).float().mean():.2%}")
    else:
        # Some MoE layers return tuples (output, router_logits)
        print(f"  Layer {i:2d} MoE MLP: type={type(m)}, len={len(m) if hasattr(m, '__len__') else 'N/A'}")
