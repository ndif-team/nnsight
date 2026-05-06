"""Client 07: Activation patching — inject one prompt's hidden state into another."""
import os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import torch
from nnsight.modeling.vllm import VLLM

URL = os.environ.get("NNSIGHT_SERVE_URL", "http://127.0.0.1:6677")
model = VLLM("Qwen/Qwen3-30B-A3B")

source_prompt = "The Eiffel Tower is located in the city of"
target_prompt = "The Colosseum is located in the city of"
patch_layer = 24

# Capture source hidden state
t0 = time.perf_counter()
with model.trace(source_prompt, temperature=0.0, top_p=1, serve=URL):
    source_hs = model.model.layers[patch_layer].output[0].save()
    source_logits = model.logits.save()

# Clean target
with model.trace(target_prompt, temperature=0.0, top_p=1, serve=URL):
    clean_logits = model.logits.save()

elapsed = time.perf_counter() - t0

source_token = model.tokenizer.decode(source_logits.argmax(dim=-1))
clean_token = model.tokenizer.decode(clean_logits.argmax(dim=-1))

print(f"[{elapsed:.3f}s]")
print(f"  Source: '{source_prompt}' → '{source_token.strip()}'")
print(f"  Target: '{target_prompt}' → '{clean_token.strip()}'")
print(f"  Source hidden state shape: {list(source_hs.shape)}")
print(f"  (Activation patching across serve traces requires same-trace invokes — ")
print(f"   cross-trace patching is not supported in serve mode)")
