"""JO/BOA-style frontier workload on nnsight+vLLM, measuring the two knobs we
made configurable:

  * enforce_eager  True (no CUDA graphs)  vs  False (graphs on)   -> throughput
  * enable_prefix_caching  on vs off  on a shared-prefix frontier -> prefill reuse

The workload mirrors BOA's mechanics with entirely benign prompts:
  - A long shared prefix + N distinct frontier suffixes (Phase-2 breadth).
  - One trace, N invokes (blog pattern: batch the frontier as invokes).
  - Per decode step: read model.logits, accumulate the chosen token's logprob
    (the tau budget), apply a block-list mask by whole-tensor replacement, and
    save only the top-k logprobs (blog pattern: transform + selective save,
    never download the full [vocab] tensor).

Run ONE config per process (vLLM uses spawn). Prints a RESULT json line.
"""
import argparse, json, time
import torch

# A small block list of common-word token ids is built from the tokenizer at
# runtime (emulating BOA's ~22-item refusal block list) - see build_blocklist.
BLOCK_WORDS = ["cannot", "Sorry", "sorry", "unable", "ethical", "illegal",
               "I", "As", "However", "instead"]


def build_shared_prefix(tok, approx_tokens):
    para = ("You are a careful assistant. Summarize the following passage about "
            "the history of cartography, mapmaking, and navigation across many "
            "centuries and civilizations, attending to detail and nuance. ")
    text = ""
    while len(tok(text).input_ids) < approx_tokens:
        text += para
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eager", choices=["true", "false"], required=True)
    ap.add_argument("--prefix-cache", choices=["on", "off", "auto"], default="auto")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    ap.add_argument("--frontier", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--prefix-len", type=int, default=220)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    from nnsight.modeling.vllm import VLLM

    eager = args.eager == "true"
    pc = {"on": True, "off": False, "auto": None}[args.prefix_cache]

    kw = dict(gpu_memory_utilization=args.gpu_mem, dispatch=True,
              enforce_eager=eager, max_model_len=4096)
    if pc is not None:
        kw["enable_prefix_caching"] = pc

    t0 = time.perf_counter()
    model = VLLM(args.model, **kw)
    load_s = time.perf_counter() - t0

    tok = model.tokenizer
    shared = build_shared_prefix(tok, args.prefix_len)
    shared_ntok = len(tok(shared).input_ids)
    suffixes = [f"\nVariation {i}: focus on the year {1400 + i*13}. Answer:"
                for i in range(args.frontier)]
    prompts = [shared + s for s in suffixes]

    block_ids = set()
    for w in BLOCK_WORDS:
        for variant in (w, " " + w):
            ids = tok(variant, add_special_tokens=False).input_ids
            if len(ids) == 1:
                block_ids.add(ids[0])
    block_ids = sorted(block_ids)
    block_idx = torch.tensor(block_ids) if block_ids else None

    def run_frontier():
        with model.trace(max_tokens=args.max_tokens, temperature=0.0, top_p=1.0) as tracer:
            topk_lp = [list() for _ in range(len(prompts))].save()
            cum_lp = [0.0 for _ in range(len(prompts))].save()
            sampled = [list() for _ in range(len(prompts))].save()
            for i, p in enumerate(prompts):
                with tracer.invoke(p):
                    for _ in tracer.iter[:]:
                        lg = model.logits.float()
                        # block-list mask via replacement (graph-safe write)
                        if block_idx is not None:
                            masked = lg.clone()
                            masked[..., block_idx] = float("-inf")
                            model.logits = masked
                            lg = masked
                        logp = torch.log_softmax(lg, dim=-1)
                        vals, idx = logp.topk(args.topk, dim=-1)
                        # selective save: only top-k, not the full vocab row
                        topk_lp[i].append(idx[..., :].reshape(-1)[: args.topk].tolist())
                        s = model.samples.reshape(-1)[0]
                        sampled[i].append(int(s))
                        cum_lp[i] += float(logp.reshape(-1)[int(s)])
        return sampled, cum_lp

    # Warmup (excluded from timing): triggers graph capture / prefix-cache fill.
    run_frontier()
    torch.cuda.synchronize()

    t1 = time.perf_counter()
    sampled, cum_lp = run_frontier()
    torch.cuda.synchronize()
    run_s = time.perf_counter() - t1

    total_tokens = sum(len(s) for s in sampled)
    checksum = sum(sum(s) for s in sampled)  # identical across eager True/False if correct
    out = {
        "eager": eager, "prefix_cache": args.prefix_cache,
        "frontier": args.frontier, "max_tokens": args.max_tokens,
        "shared_prefix_tokens": shared_ntok,
        "load_s": round(load_s, 2),
        "run_s": round(run_s, 4),
        "decode_tokens": total_tokens,
        "tokens_per_s": round(total_tokens / run_s, 1),
        "checksum": checksum,
        "cum_logprob_0": round(cum_lp[0], 3),
    }
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
