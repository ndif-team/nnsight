"""Where the J-lens demo's wall-clock goes. Times, on the Megatron backend:

  A. bare module forward (no nnsight)
  B. nnsight trace, forward only
  C. the demo shape: trace + 16 `with loss.backward():` contexts
  D. trace + 16 bare torch.autograd.grad calls (no backward contexts)

C - D isolates the per-seed BackwardTracer cost; D - B the bare autograd cost;
B - A the trace overhead; A the kernel-launch floor.

Run: CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src python tests/manual/jlens_timing.py
"""

import time

import torch

from nnsight.modeling.megatron import MegatronLM

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The quick brown fox jumps over the lazy dog"
LAYERS = [6, 12, 18]
SEEDS = 16
REPS = 5


def first(x):
    return x[0] if isinstance(x, tuple) else x


def timed(fn):
    fn()  # warmup
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(REPS):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / REPS


def main():
    mm = MegatronLM(REPO, dispatch=True, dtype=torch.float32)
    mm._module.requires_grad_(False)
    enc = mm.tokenizer(PROMPT, return_tensors="pt").to("cuda")
    module = mm._module
    directions = torch.nn.functional.normalize(torch.randn(SEEDS, 896, device="cuda"), dim=-1)

    def bare_forward():
        with torch.no_grad():
            module(**enc)

    def trace_forward():
        with mm.trace(PROMPT):
            out = mm.output.save()

    def demo_shape():
        with mm.trace(PROMPT):
            taps = {}
            for l in LAYERS:
                h = first(mm.gpt.decoder.layers[l].output)
                if not h.requires_grad:
                    h.requires_grad_(True)
                taps[l] = h
            h_final = first(mm.gpt.decoder.layers[-1].output)
            for k in range(SEEDS):
                loss = (h_final * directions[k]).sum()
                with loss.backward(retain_graph=True):
                    grads = {l: taps[l].grad.save() for l in reversed(LAYERS)}

    def bare_autograd():
        with mm.trace(PROMPT):
            taps = []
            for l in LAYERS:
                h = first(mm.gpt.decoder.layers[l].output)
                if not h.requires_grad:
                    h.requires_grad_(True)
                taps.append(h)
            h_final = first(mm.gpt.decoder.layers[-1].output)
            all_grads = []
            for k in range(SEEDS):
                loss = (h_final * directions[k]).sum()
                all_grads.append(torch.autograd.grad(loss, taps, retain_graph=True))

    a = timed(bare_forward)
    b = timed(trace_forward)
    c = timed(demo_shape)
    d = timed(bare_autograd)

    print(f"A bare forward (no nnsight):            {a*1000:8.1f} ms")
    print(f"B trace, forward only:                  {b*1000:8.1f} ms   (trace overhead {(b-a)*1000:.1f} ms)")
    print(f"D trace + {SEEDS} bare autograd.grad:        {d*1000:8.1f} ms   ({(d-b)/SEEDS*1000:.1f} ms per bare backward)")
    print(f"C trace + {SEEDS} backward contexts (demo):  {c*1000:8.1f} ms   ({(c-d)/SEEDS*1000:.1f} ms per-seed BackwardTracer overhead)")


if __name__ == "__main__":
    main()
