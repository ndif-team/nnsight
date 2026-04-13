"""Client 06: Non-blocking concurrent requests."""
import os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

from nnsight.modeling.vllm import VLLM

URL = os.environ.get("NNSIGHT_SERVE_URL", "http://127.0.0.1:6677")
model = VLLM("Qwen/Qwen3-30B-A3B")

prompts = [
    "Einstein's theory of relativity states that",
    "Quantum mechanics describes the behavior of",
    "The speed of light in vacuum is approximately",
    "Newton's first law of motion says that",
]

# Warmup
with model.trace(prompts[0], serve=URL):
    model.logits.output.save()

# Sequential baseline
t0 = time.perf_counter()
for p in prompts:
    with model.trace(p, temperature=0.0, top_p=1, serve=URL):
        model.logits.output.save()
seq_time = time.perf_counter() - t0

# Non-blocking concurrent
tracers = []
t0 = time.perf_counter()

with model.trace(prompts[0], temperature=0.0, top_p=1, serve=URL, blocking=False) as t1:
    l1 = model.logits.output.save()
with model.trace(prompts[1], temperature=0.0, top_p=1, serve=URL, blocking=False) as t2:
    l2 = model.logits.output.save()
with model.trace(prompts[2], temperature=0.0, top_p=1, serve=URL, blocking=False) as t3:
    l3 = model.logits.output.save()
with model.trace(prompts[3], temperature=0.0, top_p=1, serve=URL, blocking=False) as t4:
    l4 = model.logits.output.save()

saves = [t.collect(timeout=60) for t in [t1, t2, t3, t4]]
conc_time = time.perf_counter() - t0

print(f"Sequential: {seq_time:.3f}s")
print(f"Concurrent: {conc_time:.3f}s")
print(f"Speedup:    {seq_time / conc_time:.2f}x")
print()
for prompt, s in zip(prompts, saves):
    key = list(s.keys())[0]
    token = model.tokenizer.decode(s[key].argmax(dim=-1))
    print(f"  '{prompt}' → '{token.strip()}'")
