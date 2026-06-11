"""Parallelism equivalence matrix for nnsight + vLLM.

Compares every (TP, PP) config against the single-GPU (1,1) oracle, per model,
across a battery of scenarios. Runs each config in an isolated subprocess
(vLLM cannot re-init distributed state in-process). Layer paths and indices
are discovered at runtime from the model tree — nothing is hardcoded to a
naming convention.

Modes:
    Orchestrator (no --worker):
        python run_equivalence_matrix.py \
            --gpus 1,2,3,4 \
            --models gpt2,facebook/opt-125m,... \
            --configs 1x1,2x1,1x2,2x2 \
            --out-dir /tmp/eqv

    Worker (internal):
        python run_equivalence_matrix.py --worker \
            --model gpt2 --tp 1 --pp 1 --out /tmp/eqv/gpt2__1x1.pt
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


# --------------------------------------------------------------------------- #
# Runtime model-tree discovery
# --------------------------------------------------------------------------- #

# Names that decoder blocks typically have under them. We use the presence of
# at least one attention-like + one mlp-like child as the signature of a
# transformer decoder block, then pick the ModuleList of such blocks.
_ATTN_NAMES = {"attn", "self_attn", "attention", "self_attention"}
_MLP_NAMES = {"mlp", "feed_forward", "ffn", "fc", "block_sparse_moe", "moe"}


def discover_layers_path(underlying: torch.nn.Module) -> Tuple[str, int]:
    """Return (dotted attribute path, num_layers) of the decoder ModuleList.

    Walks ``named_modules()`` and picks the longest ``ModuleList`` whose first
    element has attention-like and mlp-like children. Family-agnostic; works
    for ``transformer.h`` (GPT2/Bloom), ``model.layers`` (Llama/Qwen/Gemma2),
    ``model.decoder.layers`` (OPT), ``gpt_neox.layers`` (Pythia), and any
    other architecture that exposes blocks as a ModuleList.
    """
    candidates: List[Tuple[str, int]] = []
    for name, mod in underlying.named_modules():
        if not isinstance(mod, torch.nn.ModuleList) or len(mod) < 2:
            continue
        first = mod[0]
        children = {n for n, _ in first.named_children()}
        if children & _ATTN_NAMES and children & _MLP_NAMES:
            candidates.append((name, len(mod)))

    if not candidates:
        # Fallback: pick the longest ModuleList anywhere, on the assumption
        # that decoder blocks are usually the longest list in a causal LM.
        for name, mod in underlying.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 2:
                candidates.append((name, len(mod)))

    if not candidates:
        raise RuntimeError(
            "Could not locate a decoder ModuleList in the model tree."
        )
    name, n = max(candidates, key=lambda kv: kv[1])
    return name, n


def resolve_envoy(envoy, dotted: str):
    """Navigate from a root envoy down a dotted attribute path."""
    cur = envoy
    for part in dotted.split("."):
        cur = getattr(cur, part)
    return cur


# --------------------------------------------------------------------------- #
# Scenario battery
#
# Every scenario takes (model, layers_envoy, num_layers, prompt) and returns
# a dict whose values are (tensors | scalars | strings). Tensors go to CPU
# float32 so the comparator can be numerically robust across dtypes.
# --------------------------------------------------------------------------- #

PROMPTS = [
    "The Eiffel Tower is in the city of",
    "In a hole in the ground there lived",
]


def _hidden(out):
    """Extract the hidden activation from a layer's output.

    Layer outputs come in three shapes across vLLM implementations:

    * Bare ``torch.Tensor`` (e.g. vLLM's GPT2 blocks),
    * ``(hidden, residual, ...)`` tuple (e.g. Qwen2, Llama decoder blocks),
    * ``LazyRemoteTensor`` wrapping either of the above on a non-owning PP
      rank.

    ``isinstance(out, tuple)`` is unreliable on the lazy proxy — it returns
    ``False`` even when the materialized value is a tuple. Probing tensor-ness
    first and indexing ``[0]`` on everything else matches the codebase
    convention (see ``tests/vllm/pp/manual/_pp_pull_worker.py``): owning-rank tuples index
    cleanly, and ``LazyRemoteTensor[0]`` returns a deferred child lazy that
    pulls the first tuple element on materialization.
    """
    if isinstance(out, torch.Tensor):
        return out
    return out[0]


def _to_cpu(x):
    return x.detach().cpu().float() if isinstance(x, torch.Tensor) else x


def scenario_logits(model, layers, n, prompt):
    with model.trace(prompt, temperature=0.0, top_p=1):
        logits = model.logits.save()
    logits = _to_cpu(logits)
    return {
        "logits": logits,
        "argmax": int(logits.argmax(dim=-1).flatten()[-1].item()),
    }


def scenario_early(model, layers, n, prompt):
    with model.trace(prompt, temperature=0.0, top_p=1):
        h = _hidden(layers[0].output).save()
    return {"hidden": _to_cpu(h)}


def scenario_mid(model, layers, n, prompt):
    idx = n // 2
    with model.trace(prompt, temperature=0.0, top_p=1):
        h = _hidden(layers[idx].output).save()
    return {"hidden": _to_cpu(h), "layer_idx": idx}


def scenario_late(model, layers, n, prompt):
    with model.trace(prompt, temperature=0.0, top_p=1):
        h = _hidden(layers[n - 1].output).save()
    return {"hidden": _to_cpu(h)}


def scenario_both_stages(model, layers, n, prompt):
    """First and last layer in one trace — both PP stages exercised together."""
    with model.trace(prompt, temperature=0.0, top_p=1):
        h0 = _hidden(layers[0].output).save()
        hn = _hidden(layers[n - 1].output).save()
        logits = model.logits.save()
    return {
        "h0": _to_cpu(h0),
        "hn": _to_cpu(hn),
        "logits": _to_cpu(logits),
    }


def scenario_ablation_mid(model, layers, n, prompt):
    """Zero out a mid-layer's output and observe how logits change.

    Tests that interventions written cross-stage (mid layer may be on stage 1
    under PP=2) reach the actual forward computation. The intervention must
    visibly change the argmax relative to the un-ablated logits saved here
    in a separate trace — but the comparator only checks that the ablated
    logits match between configs (oracle vs parallel), not that they differ
    from baseline. That keeps the comparison family-agnostic.
    """
    idx = n // 2

    # Baseline + ablated in two separate traces so the ablation doesn't
    # contaminate the baseline (vLLM doesn't support save+replace on the
    # same tensor in one trace cleanly across families).
    with model.trace(prompt, temperature=0.0, top_p=1):
        baseline = model.logits.save()

    with model.trace(prompt, temperature=0.0, top_p=1):
        out = layers[idx].output
        h = _hidden(out)
        zeroed = h * 0
        # Write back in the same shape (tuple or tensor).
        if isinstance(out, tuple):
            new = (zeroed,) + tuple(out[1:])
            layers[idx].output = new
        else:
            layers[idx].output = zeroed
        ablated = model.logits.save()

    baseline = _to_cpu(baseline)
    ablated = _to_cpu(ablated)
    return {
        "baseline_logits": baseline,
        "ablated_logits": ablated,
        "baseline_argmax": int(baseline.argmax(dim=-1).flatten()[-1].item()),
        "ablated_argmax": int(ablated.argmax(dim=-1).flatten()[-1].item()),
        "layer_idx": idx,
    }


def scenario_multigen(model, layers, n, prompt, max_tokens: int = 3):
    """Multi-token generation, per-step logits saved via tracer.iter[:N]."""
    with model.trace(prompt, temperature=0.0, top_p=1, max_tokens=max_tokens) as tracer:
        logit_list = list().save()
        for _ in tracer.iter[0:max_tokens]:
            logit_list.append(model.logits)
    logits = torch.stack([_to_cpu(l) for l in logit_list])
    argmaxes = [int(l.argmax(dim=-1).flatten()[-1].item()) for l in logits]
    return {
        "logits": logits,
        "argmaxes": argmaxes,
        "num_steps": len(logit_list),
    }


def scenario_batched(model, layers, n, prompt):
    """Two prompts in one trace via tracer.invoke().

    Saved values are bound after the outer ``with`` exits — touching them
    inside the outer ``with`` body would hit UnboundLocalError because the
    invokes execute deferred on worker threads.
    """
    with model.trace(temperature=0.0, top_p=1) as tracer:
        with tracer.invoke(PROMPTS[0]):
            l0 = model.logits.save()
        with tracer.invoke(PROMPTS[1]):
            l1 = model.logits.save()
    l0c, l1c = _to_cpu(l0), _to_cpu(l1)
    return {
        "logits_a": l0c,
        "logits_b": l1c,
        "argmax_a": int(l0c.argmax(dim=-1).flatten()[-1].item()),
        "argmax_b": int(l1c.argmax(dim=-1).flatten()[-1].item()),
    }


def scenario_cross_modify(model, layers, n, prompt):
    """Read an early layer, use its mean to perturb a late layer, save logits.

    Under PP=2 this is a cross-stage read+write (early on stage 0, late on
    stage 1). Under (1,1) it's a within-process intervention. The comparator
    asserts oracle == parallel.
    """
    with model.trace(prompt, temperature=0.0, top_p=1):
        h_early = _hidden(layers[0].output)
        delta = h_early.mean(dim=0, keepdim=True) * 0.01
        out_late = layers[n - 1].output
        h_late = _hidden(out_late)
        new_h = h_late + delta
        if isinstance(out_late, tuple):
            layers[n - 1].output = (new_h,) + tuple(out_late[1:])
        else:
            layers[n - 1].output = new_h
        logits = model.logits.save()
    logits = _to_cpu(logits)
    return {
        "logits": logits,
        "argmax": int(logits.argmax(dim=-1).flatten()[-1].item()),
    }


SCENARIOS = {
    "logits": scenario_logits,
    "early": scenario_early,
    "mid": scenario_mid,
    "late": scenario_late,
    "both_stages": scenario_both_stages,
    "ablation_mid": scenario_ablation_mid,
    "multigen": scenario_multigen,
    "batched": scenario_batched,
    "cross_modify": scenario_cross_modify,
}


# --------------------------------------------------------------------------- #
# Worker mode — runs one (model, tp, pp) and dumps a results dict
# --------------------------------------------------------------------------- #

def worker_main(args):
    """Instantiate VLLM and run the full scenario battery."""
    import nnsight
    # Surface real trace-body exceptions instead of letting them get masked
    # as UnboundLocalError when saved locals never get bound.
    nnsight.CONFIG.APP.DEBUG = True
    from nnsight.modeling.vllm import VLLM

    kwargs: Dict[str, Any] = {
        "gpu_memory_utilization": float(args.gpu_mem),
        "dispatch": True,
    }
    if args.tp > 1:
        kwargs["tensor_parallel_size"] = args.tp
    if args.pp > 1:
        kwargs["pipeline_parallel_size"] = args.pp
    if args.dtype:
        kwargs["dtype"] = args.dtype
    if args.max_model_len:
        kwargs["max_model_len"] = args.max_model_len

    print(f"[worker] loading {args.model} tp={args.tp} pp={args.pp}", flush=True)
    t0 = time.time()
    model = VLLM(args.model, **kwargs)
    print(f"[worker] loaded in {time.time()-t0:.1f}s", flush=True)

    # Discover layer path from the underlying nn.Module — not from any
    # hardcoded family table.
    underlying = model._model
    layers_path, num_layers = discover_layers_path(underlying)
    print(f"[worker] layers_path={layers_path!r} num_layers={num_layers}", flush=True)
    layers = resolve_envoy(model, layers_path)

    results: Dict[str, Any] = {
        "model": args.model,
        "tp": args.tp,
        "pp": args.pp,
        "layers_path": layers_path,
        "num_layers": num_layers,
        "scenarios": {},
    }

    prompt = PROMPTS[0]
    for name in args.scenarios.split(","):
        name = name.strip()
        if name not in SCENARIOS:
            results["scenarios"][name] = {"status": "unknown_scenario"}
            continue
        print(f"[worker] running scenario={name}", flush=True)
        try:
            t = time.time()
            out = SCENARIOS[name](model, layers, num_layers, prompt)
            out["status"] = "ok"
            out["elapsed_s"] = time.time() - t
            results["scenarios"][name] = out
            print(f"[worker]   ok ({out['elapsed_s']:.2f}s)", flush=True)
        except Exception as e:
            results["scenarios"][name] = {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            print(f"[worker]   FAILED: {e}", flush=True)

    torch.save(results, args.out)
    print(f"[worker] wrote {args.out}", flush=True)


# --------------------------------------------------------------------------- #
# Comparator
# --------------------------------------------------------------------------- #

@dataclass
class Tolerance:
    cosine: float
    max_abs: Optional[float] = None
    argmax_exact: bool = True


# Per-config tolerance — TP reorders matmul reductions so hidden states
# drift; PP alone should be near-exact. (2,2) is the looser of the two.
# argmax_exact is False under TP because BF16 reduction-order noise (~0.5
# on small models like gpt2) can flip closely-packed top tokens — that is
# inherent to the kernel, not a bug. Argmax flips are still REPORTED in
# the per-scenario detail line for visibility.
TOL = {
    (1, 1): Tolerance(cosine=1.0 - 1e-6, argmax_exact=True),
    (2, 1): Tolerance(cosine=0.999,      argmax_exact=False),  # TP
    (1, 2): Tolerance(cosine=0.9999,     argmax_exact=True),   # PP (exact)
    (2, 2): Tolerance(cosine=0.999,      argmax_exact=False),  # TP + PP
}


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item()


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def compare_scalar(oracle, candidate, key):
    """Compare a single saved value: tensor → cosine/max_abs, int → equality,
    list → element-wise (recursing), else strict equality."""
    if isinstance(oracle, torch.Tensor) and isinstance(candidate, torch.Tensor):
        if oracle.shape != candidate.shape:
            return {"kind": "tensor", "shape_match": False,
                    "oracle_shape": list(oracle.shape),
                    "cand_shape": list(candidate.shape)}
        return {
            "kind": "tensor",
            "shape_match": True,
            "cosine": cosine(oracle, candidate),
            "max_abs": max_abs(oracle, candidate),
            "shape": list(oracle.shape),
        }
    if isinstance(oracle, list) and isinstance(candidate, list):
        if len(oracle) != len(candidate):
            return {"kind": "list", "len_match": False,
                    "oracle_len": len(oracle), "cand_len": len(candidate)}
        return {
            "kind": "list",
            "len_match": True,
            "equal": oracle == candidate,
            "oracle": oracle,
            "cand": candidate,
        }
    return {"kind": "scalar", "equal": oracle == candidate,
            "oracle": oracle, "cand": candidate}


def compare_config(oracle: dict, cand: dict, tol: Tolerance) -> dict:
    """Compare every scenario between the oracle and one candidate config."""
    report = {}
    for scen, oracle_out in oracle["scenarios"].items():
        cand_out = cand["scenarios"].get(scen)
        if cand_out is None:
            report[scen] = {"status": "missing_in_candidate"}
            continue
        if oracle_out.get("status") != "ok" or cand_out.get("status") != "ok":
            report[scen] = {
                "status": "error",
                "oracle_status": oracle_out.get("status"),
                "cand_status": cand_out.get("status"),
                "oracle_error": oracle_out.get("error"),
                "cand_error": cand_out.get("error"),
            }
            continue

        per_key = {}
        passed = True
        for key, oval in oracle_out.items():
            if key in ("status", "elapsed_s"):
                continue
            cval = cand_out.get(key)
            cmp = compare_scalar(oval, cval, key)
            per_key[key] = cmp

            # Pass criteria
            if cmp["kind"] == "tensor":
                if not cmp["shape_match"]:
                    passed = False
                elif cmp["cosine"] < tol.cosine:
                    passed = False
            elif cmp["kind"] == "list":
                if not cmp.get("len_match") or not cmp.get("equal"):
                    passed = False
            elif cmp["kind"] == "scalar":
                # Strict equality for argmax / num_steps / int scalars
                if isinstance(oval, int) and tol.argmax_exact and not cmp["equal"]:
                    passed = False
        report[scen] = {"status": "ok", "passed": passed, "keys": per_key}
    return report


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

def parse_config(s: str) -> Tuple[int, int]:
    m = re.match(r"^(\d+)x(\d+)$", s.strip())
    if not m:
        raise ValueError(f"Bad config {s!r}, expected like '2x1'")
    return int(m.group(1)), int(m.group(2))


def gpu_slice(pool: List[str], n: int) -> str:
    if n > len(pool):
        raise RuntimeError(f"Need {n} GPUs, pool has {len(pool)}")
    return ",".join(pool[:n])


def slugify(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)


def reap_workers():
    """Kill straggling vLLM workers that didn't exit with the subprocess.
    Conservative: only target our own user's lingering processes."""
    user = os.environ.get("USER", "")
    for pat in ("nnsight.modeling.vllm.workers", "vllm.v1.engine.core"):
        subprocess.run(
            ["pkill", "-9", "-u", user, "-f", pat],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


def gpu_snapshot() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return " | ".join(f"GPU{l.split(',')[0].strip()}:{l.split(',')[1].strip()}MB"
                          for l in out.strip().splitlines())
    except Exception as e:
        return f"<nvidia-smi failed: {e}>"


def run_one(model: str, tp: int, pp: int, gpus: List[str], out_dir: str,
            scenarios: str, gpu_mem: float, max_model_len: Optional[int],
            dtype: Optional[str], timeout: int) -> Optional[str]:
    n_gpu = tp * pp
    visible = gpu_slice(gpus, n_gpu)
    out_path = os.path.join(out_dir, f"{slugify(model)}__{tp}x{pp}.pt")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["MASTER_PORT"] = str(29700 + hash((model, tp, pp)) % 1000)
    env["HF_HUB_CACHE"] = env.get("HF_HUB_CACHE", "/disk/u/models")

    cmd = [
        "conda", "run", "--no-capture-output", "-n", "ndif-dev", "python",
        __file__, "--worker",
        "--model", model, "--tp", str(tp), "--pp", str(pp),
        "--out", out_path, "--scenarios", scenarios,
        "--gpu-mem", str(gpu_mem),
    ]
    if max_model_len:
        cmd += ["--max-model-len", str(max_model_len)]
    if dtype:
        cmd += ["--dtype", dtype]

    print(f"\n=== {model}  tp={tp} pp={pp}  gpus={visible} ===", flush=True)
    print(f"    pre-run GPU mem: {gpu_snapshot()}", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, env=env, timeout=timeout,
                           stdout=sys.stdout, stderr=sys.stderr)
    except subprocess.TimeoutExpired:
        print(f"    !! TIMEOUT after {timeout}s", flush=True)
        reap_workers()
        return None
    elapsed = time.time() - t0
    reap_workers()
    time.sleep(2)  # let CUDA contexts drain
    print(f"    post-run GPU mem: {gpu_snapshot()}", flush=True)
    print(f"    exit={r.returncode}  elapsed={elapsed:.1f}s", flush=True)
    if r.returncode != 0 or not os.path.exists(out_path):
        return None
    return out_path


def orchestrate(args):
    gpus = args.gpus.split(",")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    configs = [parse_config(c) for c in args.configs.split(",")]
    if (1, 1) not in configs:
        configs = [(1, 1)] + configs
    os.makedirs(args.out_dir, exist_ok=True)

    # Run every (model, config). Each in its own subprocess.
    paths: Dict[Tuple[str, int, int], Optional[str]] = {}
    for model in models:
        for tp, pp in configs:
            p = run_one(model, tp, pp, gpus, args.out_dir,
                        args.scenarios, args.gpu_mem, args.max_model_len,
                        args.dtype, args.timeout)
            paths[(model, tp, pp)] = p

    # Compare each (tp,pp)!=(1,1) against (1,1) per model.
    print("\n" + "=" * 70)
    print(" Equivalence report (vs (1,1) oracle)")
    print("=" * 70)

    summary_rows: List[Tuple[str, str, str, str, str]] = []
    for model in models:
        oracle_path = paths.get((model, 1, 1))
        if not oracle_path:
            print(f"\n[{model}] no oracle — skipping")
            continue
        oracle = torch.load(oracle_path, weights_only=False)

        for tp, pp in configs:
            if (tp, pp) == (1, 1):
                continue
            cand_path = paths.get((model, tp, pp))
            tag = f"{tp}x{pp}"
            if not cand_path:
                print(f"\n[{model}  {tag}] candidate missing")
                continue
            cand = torch.load(cand_path, weights_only=False)
            tol = TOL[(tp, pp)]
            report = compare_config(oracle, cand, tol)

            print(f"\n[{model}  {tag}]  (cosine ≥ {tol.cosine})")
            for scen, r in report.items():
                if r["status"] == "error":
                    print(f"  {scen:15s}  ERROR   oracle={r['oracle_status']} cand={r['cand_status']}")
                    summary_rows.append((model, tag, scen, "ERROR", ""))
                    continue
                if r["status"] == "missing_in_candidate":
                    print(f"  {scen:15s}  MISSING")
                    summary_rows.append((model, tag, scen, "MISSING", ""))
                    continue
                # Build a short detail string from tensor keys
                bits = []
                for k, c in r["keys"].items():
                    if c["kind"] == "tensor" and c.get("shape_match"):
                        bits.append(
                            f"{k}.cos={c['cosine']:.6f} |Δ|={c['max_abs']:.3g}"
                        )
                    elif c["kind"] == "scalar":
                        bits.append(f"{k}={'==' if c['equal'] else 'NE'}")
                    elif c["kind"] == "list":
                        bits.append(f"{k}.list={'==' if c.get('equal') else 'NE'}")
                detail = "  ".join(bits)
                verdict = "PASS" if r["passed"] else "FAIL"
                print(f"  {scen:15s}  {verdict}  {detail}")
                summary_rows.append((model, tag, scen, verdict, detail))

    print("\n" + "=" * 70)
    print(" Compact summary")
    print("=" * 70)
    print(f"{'model':35s} {'cfg':6s} {'scenario':15s} {'verdict':6s}")
    for m, cfg, scen, v, _d in summary_rows:
        m_short = m if len(m) <= 33 else m[:30] + "..."
        print(f"{m_short:35s} {cfg:6s} {scen:15s} {v:6s}")


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true",
                   help="Internal worker mode")
    p.add_argument("--model", type=str)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--out", type=str)
    p.add_argument("--scenarios", type=str,
                   default=",".join(SCENARIOS.keys()))
    p.add_argument("--gpu-mem", type=float, default=0.15)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--dtype", type=str, default=None)

    # Orchestrator-only
    p.add_argument("--gpus", type=str, default="1,2,3,4",
                   help="Comma-separated CUDA_VISIBLE_DEVICES pool")
    p.add_argument("--models", type=str,
                   # SmolLM2-135M has 9 attention heads — not TP-divisible.
                   # Llama-3.2-3B (cached) is the Llama-family slot instead.
                   default="gpt2,Qwen/Qwen2.5-0.5B,Qwen/Qwen3-0.6B,"
                           "meta-llama/Llama-3.2-3B,"
                           "facebook/opt-125m,EleutherAI/pythia-1.4b,"
                           "bigcode/tiny_starcoder_py")
    p.add_argument("--configs", type=str, default="1x1,2x1,1x2,2x2")
    p.add_argument("--out-dir", type=str, default="/tmp/eqv")
    p.add_argument("--timeout", type=int, default=600)
    return p


def main():
    args = build_argparser().parse_args()
    if args.worker:
        if not args.model or not args.out:
            print("--worker requires --model and --out", file=sys.stderr)
            sys.exit(2)
        worker_main(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
