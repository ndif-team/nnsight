"""Client 04: Multiple prompts in a single trace via invoke()."""
import os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

from nnsight.modeling.vllm import VLLM

URL = os.environ.get("NNSIGHT_SERVE_URL", "http://127.0.0.1:6677")
model = VLLM("Qwen/Qwen3-30B-A3B")

prompts = [
    "The Great Wall of China is in",
    "Mount Everest is located in",
    "The Amazon River flows through",
]

t0 = time.perf_counter()
with model.trace(temperature=0.0, top_p=1, serve=URL) as tracer:
    with tracer.invoke(prompts[0]):
        logits_0 = model.logits.save()
    with tracer.invoke(prompts[1]):
        logits_1 = model.logits.save()
    with tracer.invoke(prompts[2]):
        logits_2 = model.logits.save()
elapsed = time.perf_counter() - t0

print(f"[{elapsed:.3f}s] {len(prompts)} prompts in one trace:")
for prompt, logits in zip(prompts, [logits_0, logits_1, logits_2]):
    token = model.tokenizer.decode(logits.argmax(dim=-1))
    print(f"  '{prompt}' → '{token.strip()}'")
