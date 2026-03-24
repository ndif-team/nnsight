"""
Minimal reproduction for issue #631: Inconsistent intervention results between
tensor_parallel_size=1 and tensor_parallel_size=2.

Adapted from cwdghh's script to use Qwen2.5-0.5B-Instruct (fits on packed GPUs).

Usage:
    python tests/repro_631.py --tp 2 --runs 5
    python tests/repro_631.py --tp 1 2 --runs 5   # run both and compare
"""

import argparse
import statistics
import torch
from nnsight.modeling.vllm import VLLM


MODEL_PATH   = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT       = "After John and Mary went to the store, Mary gave a bottle of milk to"
OUTPUT_TOKEN = " John"
# Qwen2.5-0.5B: 14 heads, head_dim=64, hidden=896, 24 layers
TARGET_LAYER = 20
NUM_HEADS    = 14
START_INDEX  = 0
HEAD_DIM     = 64


@torch.inference_mode()
def run_single_tp(tp: int, runs: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  Loading model  tensor_parallel_size={tp}")
    print(f"{'='*60}")

    vllm = VLLM(
        MODEL_PATH,
        tensor_parallel_size=tp,
        dispatch=True,
        dtype=torch.float16,
        gpu_memory_utilization=0.05,
    )

    token_id = vllm.tokenizer.encode(OUTPUT_TOKEN, add_special_tokens=False)[0]
    results  = {"tp": tp, "baseline": [], "heads": {h: [] for h in range(START_INDEX, NUM_HEADS)}}

    for run_idx in range(runs):
        print(f"\n[TP={tp}] Run {run_idx + 1}/{runs}")

        with torch.no_grad():
            with vllm.trace(temperature=0.0, max_tokens=1) as tracer:
                with tracer.invoke(PROMPT):
                    baseline_logit = vllm.logits.output[-1, token_id].item().save()

        bl = float(baseline_logit)
        results["baseline"].append(bl)
        print(f"  Baseline logit: {bl:.6f}")

        with torch.no_grad():
            with vllm.trace(temperature=0.0, max_tokens=1) as tracer:
                saved = dict().save()
                for head in range(START_INDEX, NUM_HEADS):
                    with tracer.invoke(PROMPT):
                        head_in = (
                            vllm.model.layers[TARGET_LAYER]
                            .self_attn.o_proj.input.clone()
                        )
                        s, e = head * HEAD_DIM, (head + 1) * HEAD_DIM
                        head_in[:, s:e] = 0
                        vllm.model.layers[TARGET_LAYER].self_attn.o_proj.input = head_in
                        saved[head] = vllm.logits.output[-1, token_id].item().save()

        for head, sv in saved.items():
            v = float(sv)
            results["heads"][head].append(v)
            diff = bl - v
            print(f"  head={head:2d}  logit={v:.6f}  diff={diff:+.6f}")

    del vllm
    torch.cuda.empty_cache()

    return results


def stats(values):
    mean = statistics.mean(values)
    std  = statistics.pstdev(values)
    return mean, std


def print_results(all_results):
    if len(all_results) >= 2:
        tp_list = sorted(all_results.keys())
        res1, res2 = all_results[tp_list[0]], all_results[tp_list[1]]
        tp1, tp2 = res1["tp"], res2["tp"]

        bl1_mean, bl1_std = stats(res1["baseline"])
        bl2_mean, bl2_std = stats(res2["baseline"])

        print(f"\n{'─'*80}")
        print(f"  Baseline logit  (token='{OUTPUT_TOKEN}')")
        print(f"  TP={tp1}: mean={bl1_mean:.6f}  std={bl1_std:.6f}")
        print(f"  TP={tp2}: mean={bl2_mean:.6f}  std={bl2_std:.6f}  delta={bl2_mean - bl1_mean:+.6f}")
        print(f"{'─'*80}")

        print(f"\n  Intervention logits  (layer={TARGET_LAYER}, one head zeroed per row)")
        print(f"  {'Head':>4}  {'TP='+str(tp1)+' mean':>12}  {'std':>10}  {'TP='+str(tp2)+' mean':>12}  {'std':>10}  {'Δ mean':>10}  {'Unstable?':>10}")

        unstable_heads = []
        for head in range(START_INDEX, NUM_HEADS):
            m1, s1 = stats(res1["heads"][head])
            m2, s2 = stats(res2["heads"][head])
            flag = "***" if s2 > 1e-4 else ""
            if s2 > 1e-4:
                unstable_heads.append(head)
            print(f"  {head:>4}  {m1:>12.6f}  {s1:>10.6f}  {m2:>12.6f}  {s2:>10.6f}  {m2-m1:>+10.6f}  {flag:>10}")

        print(f"\n  TP={tp2} unstable heads (std > 1e-4): {unstable_heads if unstable_heads else 'none'}")
        max_delta = max(abs(stats(res2["heads"][h])[0] - stats(res1["heads"][h])[0])
                        for h in range(START_INDEX, NUM_HEADS))
        print(f"  Max |mean(TP={tp2}) - mean(TP={tp1})| across heads: {max_delta:.6f}")
    else:
        tp = list(all_results.keys())[0]
        res = all_results[tp]
        bl_mean, bl_std = stats(res["baseline"])
        print(f"\nBaseline: mean={bl_mean:.6f}  std={bl_std:.6f}")

        unstable_heads = []
        for head in range(START_INDEX, NUM_HEADS):
            m, s = stats(res["heads"][head])
            flag = "***" if s > 1e-4 else ""
            if s > 1e-4:
                unstable_heads.append(head)
            print(f"  head={head:>2}  mean={m:.6f}  std={s:.6f}  {flag}")

        print(f"\n  Unstable heads (std > 1e-4): {unstable_heads if unstable_heads else 'none'}")


def main():
    parser = argparse.ArgumentParser(description="TP inconsistency reproducer (issue #631)")
    parser.add_argument("--tp", type=int, nargs="+", default=[2],
                        help="tensor_parallel_size values to test")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of repeat runs")
    args = parser.parse_args()

    all_results = {}
    for tp in args.tp:
        all_results[tp] = run_single_tp(tp, runs=args.runs)

    print_results(all_results)


if __name__ == "__main__":
    main()
