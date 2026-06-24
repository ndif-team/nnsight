"""
Integration tests for Pipeline Parallelism (PP) in NNsight's vLLM integration.

Compares PP=2 results against PP=1 (baseline) for GPT-2 on 2 GPUs.
Bitwise equivalence is not expected due to numerical differences in
PP execution. Tests verify deviations are acceptable: same top-k
predictions, cosine similarity > 0.99.

GPT-2 has 12 layers. With PP=2:
  Stage 0 (rank 0): layers 0-5
  Stage 1 (rank 1): layers 6-11 + ln_f + lm_head

Each test scenario runs PP=1 and PP=2 in separate subprocesses to avoid
distributed environment conflicts and GPU memory contention. Results are
compared in the parent process.

STATUS: PP=2 cross-stage reads — final logits, early/late hidden states, and
multi-token generation — match PP=1 and pass. (The mediator-deserialization
failure once recorded here, where serialized mediators referenced module paths
absent on non-owning ranks, has since been fixed.) The one remaining xfail is
the cross-stage *write* test, and not for that old reason: its scenario does an
in-place update `model.transformer.h[8].output[0][:] = h2`, which vLLM forbids
on inference-mode tensors — PP=1 raises "Inplace update to inference tensor
outside InferenceMode", and PP=2 does not error but silently drops the write
(unperturbed output). Isolating whether a cross-stage write actually applies on
PP=2 needs the documented replacement pattern; see CROSS_STAGE_WRITE_XFAIL_REASON.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch
import torch.nn.functional as F


# =============================================================================
# GPU detection: find 3 free GPUs (1 for PP=1, 2 for PP=2)
# =============================================================================

def _find_free_gpus(min_free_mib=4000):
    """Return indices of GPUs with at least min_free_mib free memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        free = []
        for line in result.stdout.strip().split("\n"):
            parts = line.split(",")
            idx, free_mib = int(parts[0].strip()), int(parts[1].strip())
            if free_mib >= min_free_mib:
                free.append(idx)
        return free
    except Exception:
        return []


FREE_GPUS = _find_free_gpus(min_free_mib=4000)

if len(FREE_GPUS) < 3:
    pytest.skip(
        f"PP integration tests need 3 free GPUs (1 for PP=1, 2 for PP=2), "
        f"found {len(FREE_GPUS)} free: {FREE_GPUS}",
        allow_module_level=True,
    )

# Assign GPUs: PP=1 gets one GPU, PP=2 gets two different GPUs
GPU_PP1 = str(FREE_GPUS[0])
GPU_PP2 = f"{FREE_GPUS[1]},{FREE_GPUS[2]}"

# Use the prompt from existing tests that expects " Paris"
PROMPT = "The Eiffel Tower is located in the city of"
GEN_PROMPT = "Madison Square Garden is located in the city of"


def cosine_sim(a_list, b_list):
    """Compute cosine similarity between two flat float lists."""
    a = torch.tensor(a_list, dtype=torch.float32).flatten()
    b = torch.tensor(b_list, dtype=torch.float32).flatten()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# =============================================================================
# Subprocess runner
# =============================================================================

WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "manual", "_pp_worker.py")
# tests/vllm/pp/ -> repo root is three levels up.
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)

# Expected-failure reason for the cross-stage write test (the one remaining xfail).
CROSS_STAGE_WRITE_XFAIL_REASON = (
    "Cross-stage write uses an in-place update "
    "(model.transformer.h[8].output[0][:] = h2), which vLLM forbids on its "
    "inference-mode output tensors: PP=1 raises 'Inplace update to inference "
    "tensor outside InferenceMode is not allowed'. On PP=2 the same op does not "
    "error but is dropped (output is the unperturbed ' Paris'). Rework with the "
    "documented replacement pattern to isolate whether a cross-stage write "
    "actually applies on PP=2."
)


def _run_worker(cuda_visible_devices, scenario, extra_args=None):
    """Run a test scenario in a subprocess with specified CUDA_VISIBLE_DEVICES.

    Returns parsed JSON output from the worker script.
    Raises RuntimeError on failure.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = f.name

    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        cmd = [
            sys.executable,
            WORKER_SCRIPT,
            scenario,
            "--output", output_path,
        ]
        if extra_args:
            cmd.extend(extra_args)

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, env=env,
            cwd=REPO_ROOT,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Worker failed (scenario={scenario}, CUDA={cuda_visible_devices}, rc={result.returncode}):\n"
                f"STDERR (last 4000 chars):\n{result.stderr[-4000:] if result.stderr else '(empty)'}"
            )

        with open(output_path, "r") as f:
            data = json.load(f)

        if data.get("status") == "error":
            raise RuntimeError(
                f"Worker error (scenario={scenario}, PP CUDA={cuda_visible_devices}):\n"
                f"{data.get('error', 'unknown')}\n"
                f"{data.get('traceback', '')}"
            )

        return data

    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


# =============================================================================
# Test A: Basic logits comparison
# =============================================================================


class TestBasicLogits:
    """Compare final logits between PP=1 and PP=2."""

    def test_pp1_baseline(self):
        """PP=1 produces expected result (sanity check)."""
        pp1 = _run_worker(GPU_PP1, "logits", ["--pp", "1", "--prompt", PROMPT])
        assert pp1["top_token"] == " Paris", (
            f"PP=1 baseline failed: expected ' Paris', got {pp1['top_token']!r}"
        )

    def test_logits_same_argmax_and_cosine(self):
        """Top-1 prediction should be identical, cosine sim > 0.99."""
        pp1 = _run_worker(GPU_PP1, "logits", ["--pp", "1", "--prompt", PROMPT])
        pp2 = _run_worker(GPU_PP2, "logits", ["--pp", "2", "--prompt", PROMPT])

        print(f"\n[Test A] PP=1 top-1: {pp1['top_token']!r} (id={pp1['argmax']})")
        print(f"[Test A] PP=2 top-1: {pp2['top_token']!r} (id={pp2['argmax']})")

        assert pp1["argmax"] == pp2["argmax"], (
            f"Argmax mismatch: PP=1={pp1['top_token']!r} ({pp1['argmax']}), "
            f"PP=2={pp2['top_token']!r} ({pp2['argmax']})"
        )

        sim = cosine_sim(pp1["logits"], pp2["logits"])
        print(f"[Test A] Logit cosine similarity: {sim:.6f}")
        assert sim > 0.99, f"Cosine similarity too low: {sim:.6f}"


# =============================================================================
# Test B: Hidden state from early layer (layer 0, on stage 0 for both)
# =============================================================================


class TestEarlyLayerHidden:
    """Compare hidden states from layer 0 (always on stage 0)."""

    def test_layer0_cosine_similarity(self):
        """Layer 0 hidden states should be near-identical."""
        pp1 = _run_worker(GPU_PP1, "hidden_only", ["--pp", "1", "--prompt", PROMPT, "--layer", "0"])
        pp2 = _run_worker(GPU_PP2, "hidden_only", ["--pp", "2", "--prompt", PROMPT, "--layer", "0"])

        print(f"\n[Test B] PP=1 layer 0 shape: {pp1['shape']}")
        print(f"[Test B] PP=2 layer 0 shape: {pp2['shape']}")

        sim = cosine_sim(pp1["hidden"], pp2["hidden"])
        print(f"[Test B] Layer 0 cosine similarity: {sim:.6f}")
        assert sim > 0.99, f"Cosine similarity too low: {sim:.6f}"


# =============================================================================
# Test C: Hidden state from late layer (layer 11, on stage 1 for PP=2)
# =============================================================================


class TestLateLayerHidden:
    """Compare hidden states from layer 11 (stage 1 for PP=2)."""

    def test_layer11_cosine_similarity(self):
        """Layer 11 hidden states should be highly similar."""
        pp1 = _run_worker(GPU_PP1, "hidden_only", ["--pp", "1", "--prompt", PROMPT, "--layer", "11"])
        pp2 = _run_worker(GPU_PP2, "hidden_only", ["--pp", "2", "--prompt", PROMPT, "--layer", "11"])

        print(f"\n[Test C] PP=1 layer 11 shape: {pp1['shape']}")
        print(f"[Test C] PP=2 layer 11 shape: {pp2['shape']}")

        sim = cosine_sim(pp1["hidden"], pp2["hidden"])
        print(f"[Test C] Layer 11 cosine similarity: {sim:.6f}")
        assert sim > 0.99, f"Cosine similarity too low: {sim:.6f}"


# =============================================================================
# Test D: Cross-stage write (read layer 2, write to layer 8)
# =============================================================================


class TestCrossStageWrite:
    """Test cross-stage intervention: read from layer 2, write to layer 8.

    With PP=2:
      - Layer 2 is on stage 0 (rank 0)
      - Layer 8 is on stage 1 (rank 1)
    """

    @pytest.mark.xfail(reason=CROSS_STAGE_WRITE_XFAIL_REASON, strict=True)
    def test_cross_stage_write_same_argmax(self):
        """Cross-stage write should produce same top-1 token."""
        pp1 = _run_worker(GPU_PP1, "cross_stage", ["--pp", "1", "--prompt", PROMPT])
        pp2 = _run_worker(GPU_PP2, "cross_stage", ["--pp", "2", "--prompt", PROMPT])

        print(f"\n[Test D] PP=1 cross-stage top-1: {pp1['top_token']!r}")
        print(f"[Test D] PP=2 cross-stage top-1: {pp2['top_token']!r}")

        assert pp1["argmax"] == pp2["argmax"], (
            f"Cross-stage write argmax mismatch: "
            f"PP=1={pp1['top_token']!r} ({pp1['argmax']}), "
            f"PP=2={pp2['top_token']!r} ({pp2['argmax']})"
        )


# =============================================================================
# Test E: Multi-token generation
# =============================================================================


class TestMultiTokenGeneration:
    """Compare multi-token generation between PP=1 and PP=2."""

    def test_multi_token_same_argmax(self):
        """Each generation step should produce the same top-1 token."""
        num_tokens = 3
        pp1 = _run_worker(GPU_PP1, "multigen", [
            "--pp", "1", "--prompt", GEN_PROMPT, "--max_tokens", str(num_tokens),
        ])
        pp2 = _run_worker(GPU_PP2, "multigen", [
            "--pp", "2", "--prompt", GEN_PROMPT, "--max_tokens", str(num_tokens),
        ])

        assert len(pp1["tokens"]) == num_tokens, (
            f"PP=1 produced {len(pp1['tokens'])} tokens, expected {num_tokens}"
        )
        assert len(pp2["tokens"]) == num_tokens, (
            f"PP=2 produced {len(pp2['tokens'])} tokens, expected {num_tokens}"
        )

        all_match = True
        for step in range(num_tokens):
            tok1 = pp1["tokens"][step]
            tok2 = pp2["tokens"][step]
            match = pp1["argmaxes"][step] == pp2["argmaxes"][step]
            if not match:
                all_match = False
            print(f"\n[Test E] Step {step}: PP=1={tok1!r}, PP=2={tok2!r} {'MATCH' if match else 'MISMATCH'}")

        assert all_match, (
            f"Multi-token generation mismatch. PP=1={pp1['tokens']}, PP=2={pp2['tokens']}"
        )
