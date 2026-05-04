"""Client 03: Zero out an MLP and verify the prediction changes."""
import os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

from nnsight.modeling.vllm import VLLM
import torch

URL = os.environ.get("NNSIGHT_SERVE_URL", "http://127.0.0.1:6677")
model = VLLM("Qwen/Qwen3-30B-A3B")

prompt = "The Eiffel Tower is located in the city of"

# Clean run
t0 = time.perf_counter()
with model.trace(prompt, temperature=0.0, top_p=1, serve=URL):
    clean_logits = model.logits.save()
clean_time = time.perf_counter() - t0
clean_token = model.tokenizer.decode(clean_logits.argmax(dim=-1))

# Corrupted run — zero out last 8 layers' outputs
t0 = time.perf_counter()
with model.trace(prompt, temperature=0.0, top_p=1, serve=URL):
    for layer_idx in range(40, 48):
        model.model.layers[layer_idx].output[0][:] = 0
    corrupted_logits = model.logits.save()
corrupt_time = time.perf_counter() - t0
corrupted_token = model.tokenizer.decode(corrupted_logits.argmax(dim=-1))

print(f"[{clean_time:.3f}s] Clean:     '{prompt}' → '{clean_token.strip()}'")
print(f"[{corrupt_time:.3f}s] Corrupted: '{prompt}' → '{corrupted_token.strip()}'")

if clean_token != corrupted_token:
    print("Intervention changed prediction ✓")
else:
    print("WARNING: prediction unchanged — intervention may not have taken effect")
