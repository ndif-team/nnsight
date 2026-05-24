"""Minimal gate: does a logits-only (JO/BOA-style) nnsight+vLLM trace work
with enforce_eager=False (CUDA graphs ON)?

We check both directions of the logits intervention under graphs-on:
  - READ:  saved logits are finite and the greedy argmax matches eager=True.
  - WRITE: forcing the top token's logit to -inf changes the sampled token,
           proving the post-forward logits hook fires through the graph path.

Run one config per process (vLLM uses spawn): pass --eager {true,false}.
"""
import argparse, json, sys
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eager", choices=["true", "false"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--gpu-mem", type=float, default=0.2)
    args = ap.parse_args()

    from nnsight.modeling.vllm import VLLM

    eager = args.eager == "true"
    model = VLLM(args.model, gpu_memory_utilization=args.gpu_mem,
                 dispatch=True, enforce_eager=eager,
                 max_model_len=2048)

    prompt = "The capital of France is"

    # READ: greedy next-token logits.
    with model.trace(prompt, temperature=0.0, top_p=1.0, max_tokens=1):
        lg = model.logits.save()
    lg = lg.float()
    finite = bool(torch.isfinite(lg).all().item())
    greedy_id = int(lg.argmax(dim=-1).flatten()[0].item())
    greedy_tok = model.tokenizer.decode([greedy_id])

    # WRITE: ban that same top token (BOA block-list mechanic) via whole-tensor
    # replacement (avoids the in-place fused-kernel mutation gap); the sampler
    # must now pick a different token.
    with model.trace(prompt, temperature=0.0, top_p=1.0, max_tokens=1):
        masked = model.logits.clone()
        masked[..., greedy_id] = float("-inf")
        model.logits = masked
        s = model.samples.save()
    banned_id = int(torch.as_tensor(s).flatten()[0].item())
    banned_tok = model.tokenizer.decode([banned_id])

    out = {
        "eager": eager,
        "logits_finite": finite,
        "logits_shape": list(lg.shape),
        "greedy_id": greedy_id, "greedy_tok": greedy_tok,
        "write_took_effect": banned_id != greedy_id,
        "banned_id": banned_id, "banned_tok": banned_tok,
    }
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
