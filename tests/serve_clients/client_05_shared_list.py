"""Client 05: Shared list across invokes — collect argmax tokens."""
import os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

from nnsight.modeling.vllm import VLLM

URL = os.environ.get("NNSIGHT_SERVE_URL", "http://127.0.0.1:6677")
model = VLLM("Qwen/Qwen3-30B-A3B")

prompts = [
    "Python was created by",
    "The first programming language was",
    "Linux was developed by",
    "The inventor of the World Wide Web is",
]

t0 = time.perf_counter()
with model.trace(temperature=0.0, top_p=1, serve=URL) as tracer:
    tokens = [None for _ in prompts].save()
    for i, prompt in enumerate(prompts):
        with tracer.invoke(prompt):
            tokens[i] = model.logits.output.argmax(dim=-1)
elapsed = time.perf_counter() - t0

print(f"[{elapsed:.3f}s] Shared list across {len(prompts)} invokes:")
for prompt, tok in zip(prompts, tokens):
    decoded = model.tokenizer.decode(tok)
    print(f"  '{prompt}' → '{decoded.strip()}'")
