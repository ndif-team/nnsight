"""Client 01: Basic logit capture and next-token prediction."""
import os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

from nnsight.modeling.vllm import VLLM

URL = os.environ.get("NNSIGHT_SERVE_URL", "http://127.0.0.1:6677")
model = VLLM("Qwen/Qwen3-30B-A3B")

prompts = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The largest planet in our solar system is",
]

for prompt in prompts:
    t0 = time.perf_counter()
    with model.trace(prompt, temperature=0.0, top_p=1, serve=URL):
        logits = model.logits.output.save()
    elapsed = time.perf_counter() - t0

    token = model.tokenizer.decode(logits.argmax(dim=-1))
    print(f"[{elapsed:.3f}s] '{prompt}' → '{token.strip()}'")
