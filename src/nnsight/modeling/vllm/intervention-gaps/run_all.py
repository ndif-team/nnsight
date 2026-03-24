"""
Orchestrator: runs each gap test as TWO subprocesses (vLLM + HF) on separate
GPUs and prints a comparison summary table.

Each test+backend runs in its own subprocess on its own GPU, so vLLM and HF
can run in parallel without interfering.

Usage:
    python vllm-intervention-gaps/run_all.py --vllm-gpu 2 --hf-gpu 4
    python vllm-intervention-gaps/run_all.py --vllm-gpu 2 --hf-gpu 4 --test 1_1
    python vllm-intervention-gaps/run_all.py --vllm-gpu 2  # vLLM only
    python vllm-intervention-gaps/run_all.py --hf-gpu 4    # HF only
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

TESTS = [
    ("1_1", "In-place mutation corrupts .save()"),
    ("1_2", "Decoder layer output semantics differ"),
    ("1_3", "Decoder layer input semantics differ"),
    ("1_4", "LayerNorm output: tensor vs tuple"),
    ("1_5", "LayerNorm input semantics differ"),
    ("2_1", "MLP submodule layout (merged gate_up_proj)"),
    ("2_2", "Attention submodule layout (merged qkv_proj)"),
    ("2_3", "RowParallelLinear returns tuple"),
    ("3_1", "Flat batch dimension [total_tokens, hidden]"),
    ("3_2", "PagedAttention: no attention weights"),
    ("4_1", "Gradients blocked by inference_mode"),
    ("4_2", "Source tracing into fused kernels"),
    ("4_3", "Module skip breaks fused norm"),
]

HERE = Path(__file__).parent


def run_test(test_id: str, backend: str, gpu: str, model: str) -> dict:
    """Run a single test with a specific backend in a subprocess."""
    script = HERE / f"test_{test_id}.py"
    if not script.exists():
        return {
            "id": test_id,
            "backend": backend,
            "status": "MISSING",
            "detail": f"{script.name} not found",
        }

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu

    cmd = [sys.executable, str(script), "--model", model, "--backend", backend]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, env=env
        )
    except subprocess.TimeoutExpired:
        return {
            "id": test_id,
            "backend": backend,
            "status": "TIMEOUT",
            "detail": "Exceeded 300s",
        }

    # Look for JSON result on last non-empty line of stdout
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    if lines:
        for line in reversed(lines):
            try:
                result = json.loads(line)
                result["id"] = test_id
                result["backend"] = backend
                return result
            except json.JSONDecodeError:
                continue

    return {
        "id": test_id,
        "backend": backend,
        "status": "ERROR",
        "detail": proc.stderr[-500:] if proc.stderr else "No output",
        "stdout": proc.stdout[-500:] if proc.stdout else "",
    }


def compare_results(vllm_result: dict, hf_result: dict) -> str:
    """Compare vLLM and HF results and return a gap verdict."""
    vs = vllm_result.get("status", "?")
    hs = hf_result.get("status", "?")

    if vs == "CONFIRMED" and hs == "NO_GAP":
        return "GAP CONFIRMED"
    elif vs == "CONFIRMED" and hs in ("UNEXPECTED", "UNEXPECTED_FAILURE", "UNEXPECTED_MATCH", "UNEXPECTED_MISMATCH"):
        return "GAP CONFIRMED (HF unexpected)"
    elif vs == "NOT_REPRODUCED" and hs == "NO_GAP":
        return "GAP NOT REPRODUCED"
    elif vs in ("ERROR", "TIMEOUT", "MISSING"):
        return f"vLLM {vs}"
    elif hs in ("ERROR", "TIMEOUT", "MISSING"):
        return f"HF {hs}"
    else:
        return f"vLLM={vs}, HF={hs}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-gpu", default=None, help="GPU for vLLM (omit to skip vLLM)")
    parser.add_argument("--hf-gpu", default=None, help="GPU for HF (omit to skip HF)")
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B", help="Model to test")
    parser.add_argument("--test", default=None, help="Run only this test (e.g. 1_1)")
    args = parser.parse_args()

    if args.vllm_gpu is None and args.hf_gpu is None:
        parser.error("Provide at least one of --vllm-gpu or --hf-gpu")

    tests_to_run = TESTS
    if args.test:
        tests_to_run = [(tid, desc) for tid, desc in TESTS if tid == args.test]
        if not tests_to_run:
            print(f"Unknown test: {args.test}")
            sys.exit(1)

    run_vllm = args.vllm_gpu is not None
    run_hf = args.hf_gpu is not None

    results = []
    for tid, desc in tests_to_run:
        print(f"\n{'='*70}")
        print(f"Gap {tid}: {desc}")
        print(f"{'='*70}")

        vllm_result = None
        hf_result = None

        if run_vllm:
            print(f"  Running vLLM (GPU {args.vllm_gpu})...")
            vllm_result = run_test(tid, "vllm", args.vllm_gpu, args.model)
            vs = vllm_result.get("status", "?")
            vd = vllm_result.get("detail", "")
            print(f"  vLLM -> {vs}: {vd}")

        if run_hf:
            print(f"  Running HF (GPU {args.hf_gpu})...")
            hf_result = run_test(tid, "hf", args.hf_gpu, args.model)
            hs = hf_result.get("status", "?")
            hd = hf_result.get("detail", "")
            print(f"  HF   -> {hs}: {hd}")

        if vllm_result and hf_result:
            verdict = compare_results(vllm_result, hf_result)
            print(f"  >>> {verdict}")
        elif vllm_result:
            verdict = vllm_result.get("status", "?")
        elif hf_result:
            verdict = hf_result.get("status", "?")
        else:
            verdict = "SKIPPED"

        results.append((tid, desc, vllm_result, hf_result, verdict))

    # Summary table
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")

    if run_vllm and run_hf:
        print(f"{'Gap':<6} {'vLLM':<18} {'HF':<18} {'Verdict':<22} {'Description'}")
        print(f"{'-'*6} {'-'*18} {'-'*18} {'-'*22} {'-'*40}")
        for tid, desc, vr, hr, verdict in results:
            vs = vr.get("status", "?") if vr else "-"
            hs = hr.get("status", "?") if hr else "-"
            print(f"{tid:<6} {vs:<18} {hs:<18} {verdict:<22} {desc}")
    else:
        backend_name = "VLLM" if run_vllm else "HF"
        print(f"{'Gap':<6} {backend_name:<18} {'Description'}")
        print(f"{'-'*6} {'-'*18} {'-'*40}")
        for tid, desc, vr, hr, verdict in results:
            r = vr if vr else hr
            status = r.get("status", "?") if r else "?"
            print(f"{tid:<6} {status:<18} {desc}")

    # Count stats
    if run_vllm and run_hf:
        confirmed = sum(1 for _, _, _, _, v in results if "GAP CONFIRMED" in v)
        not_reproduced = sum(1 for _, _, _, _, v in results if "NOT REPRODUCED" in v)
        other = len(results) - confirmed - not_reproduced
        print(f"\nGaps confirmed: {confirmed}/{len(results)}")
        if not_reproduced:
            print(f"Gaps not reproduced: {not_reproduced}/{len(results)}")
        if other:
            print(f"Other: {other}/{len(results)}")


if __name__ == "__main__":
    main()
