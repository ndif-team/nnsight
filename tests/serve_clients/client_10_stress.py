"""Client 10: Stress test — rapid-fire 20 blocking requests."""
import os, time, random
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import torch
from nnsight.modeling.vllm import VLLM

URL = os.environ.get("NNSIGHT_SERVE_URL", "http://127.0.0.1:6677")
model = VLLM("Qwen/Qwen3-30B-A3B")

prompts = [
    "The theory of evolution was proposed by",
    "Photosynthesis converts sunlight into",
    "The human brain contains approximately",
    "DNA stands for",
    "The periodic table was organized by",
    "Black holes are formed when",
    "The speed of sound in air is about",
    "Antibiotics were discovered by",
    "The Pythagorean theorem states that",
    "Machine learning models learn from",
    "The largest ocean on Earth is the",
    "Gravity was first described by",
    "The boiling point of water at sea level is",
    "Electrons orbit the nucleus in",
    "The mitochondria is known as the",
    "Plate tectonics explains how",
    "The Hubble telescope was launched in",
    "RNA differs from DNA in that",
    "Fibonacci numbers follow the pattern",
    "The greenhouse effect is caused by",
]

random.shuffle(prompts)
N = len(prompts)

successes = 0
failures = 0
times = []

print(f"Firing {N} requests sequentially...")
t_total = time.perf_counter()

for i, prompt in enumerate(prompts):
    t0 = time.perf_counter()
    try:
        with model.trace(prompt, temperature=0.0, top_p=1, serve=URL):
            logits = model.logits.save()
        token = model.tokenizer.decode(logits.argmax(dim=-1)).strip()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        successes += 1
        print(f"  [{i+1:2d}/{N}] {elapsed:.3f}s OK  '{prompt[:40]}...' → '{token[:20]}'")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        failures += 1
        print(f"  [{i+1:2d}/{N}] {elapsed:.3f}s ERR '{prompt[:40]}...' → {type(e).__name__}: {e}")

total_time = time.perf_counter() - t_total

print(f"\n{'='*60}")
print(f"Results: {successes}/{N} succeeded, {failures}/{N} failed")
print(f"Total:   {total_time:.2f}s")
if times:
    print(f"Latency: min={min(times):.3f}s, max={max(times):.3f}s, "
          f"avg={sum(times)/len(times):.3f}s, median={sorted(times)[len(times)//2]:.3f}s")
    print(f"Throughput: {N / total_time:.1f} req/s")
