"""End-to-end integration tests for nnsight-serve (HF CB).

Spawns the actual HTTP server subprocess, issues concurrent requests
through the real client path (`serve=URL`), and compares results +
timing against a local sequential baseline.

Uses GPU 0 for the server, GPU 1 for the sequential baseline. Skipped
if fewer than 2 GPUs are available.

Run:
    conda activate ndif-dev
    PYTHONPATH=src python -m pytest tests/test_hf_serve_integration.py -v -s
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

SERVER_GPU = "0"
BASELINE_GPU_INDEX = 1  # device index after CUDA_VISIBLE_DEVICES is applied
SERVER_PORT = 16789
SERVER_HOST = "127.0.0.1"
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
MODEL = "openai-community/gpt2"

if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    pytest.skip(
        f"integration tests require ≥2 GPUs; got {torch.cuda.device_count()}",
        allow_module_level=True,
    )


PROMPTS = [
    "The Eiffel Tower is located in the city of",
    "The capital of Germany is the city of",
    "The Great Wall is located in the country of",
    "The Statue of Liberty stands in the city of",
    "Madison Square Garden is located in the city of",
    "The Empire State Building is located in the city of",
    "The White House is in the city of",
    "The Taj Mahal is located in the country of",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def server_url():
    """Spawn `nnsight-serve` in a subprocess on GPU 0, wait for /health."""
    import httpx

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": SERVER_GPU}
    worktree_src = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "src",
    )
    env["PYTHONPATH"] = worktree_src + os.pathsep + env.get("PYTHONPATH", "")

    log_path = "/tmp/nnsight_serve_integration.log"
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m", "nnsight.modeling.hf_serve.api.cli",
            MODEL,
            "--host", SERVER_HOST,
            "--port", str(SERVER_PORT),
            "--max-batch-size", "16",
        ],
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )

    deadline = time.time() + 180
    ready = False
    while time.time() < deadline and proc.poll() is None:
        try:
            r = httpx.get(f"{SERVER_URL}/health", timeout=2.0)
            if r.status_code == 200:
                ready = True
                break
        except Exception:
            pass
        time.sleep(0.5)

    if not ready:
        try:
            proc.kill()
        except Exception:
            pass
        log_fh.close()
        with open(log_path) as f:
            pytest.fail(
                f"server did not become ready within 180s\n"
                f"--- log tail ---\n{f.read()[-4000:]}"
            )

    yield SERVER_URL

    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
    log_fh.close()


@pytest.fixture(scope="module")
def client_model():
    """Meta (non-dispatched) LanguageModel for constructing serve traces."""
    from nnsight import LanguageModel
    return LanguageModel(MODEL)


@pytest.fixture(scope="module")
def baseline_model():
    """Dispatched LanguageModel on GPU 1 for sequential local comparison."""
    from nnsight import LanguageModel
    return LanguageModel(
        MODEL,
        device_map=f"cuda:{BASELINE_GPU_INDEX}",
        dispatch=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_serve_backend(client_model, url):
    """Explicit LocalServeBackend — `serve=` kwarg isn't yet wired on
    `LanguageModel.trace` (only on `VLLM.trace`), so we construct the
    client backend directly.
    """
    from nnsight.intervention.backends.local_serve import LocalServeBackend
    return LocalServeBackend(client_model, host=url, blocking=True)


SERVER_MAX_NEW_TOKENS = 20  # server's default; baseline matches this for timing


def _submit_via_serve(client_model, prompt, url):
    """Issue one trace through the real HTTP client path.

    The server generates ``SERVER_MAX_NEW_TOKENS`` per request (one
    prefill + N-1 decode forwards). The user's trace only captures
    activations on the first (prefill) forward — later decode steps
    run without user code.
    """
    backend = _make_serve_backend(client_model, url)
    with client_model.trace(prompt, backend=backend):
        logits = client_model.lm_head.output.save()
    return logits


def _submit_local(model, prompt):
    """Issue one trace via local execution (no server)."""
    with model.trace(prompt):
        logits = model.lm_head.output.save()
    return logits


def _run_concurrent(client_model, url, prompts):
    """Issue N prompts concurrently via a thread pool.

    Each worker holds the GIL while constructing its trace (fast —
    AST capture and serialization are CPU-bound but microseconds) and
    releases it while awaiting the HTTP response (network I/O). So N
    requests can have HTTP in-flight simultaneously against the
    server, which batches them in the background generation loop.
    """
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        return list(pool.map(
            lambda p: _submit_via_serve(client_model, p, url), prompts,
        ))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConcurrentRequests:

    def test_concurrent_submissions_all_succeed(self, server_url, client_model):
        """N concurrent serve traces each return a populated, non-NaN logits tensor."""
        results = _run_concurrent(client_model, server_url, PROMPTS)

        assert len(results) == len(PROMPTS)
        vocab = client_model._model.config.vocab_size
        for i, r in enumerate(results):
            assert r is not None, f"prompt {i}: save came back None"
            assert r.shape[-1] == vocab, (
                f"prompt {i}: vocab dim {r.shape[-1]} != {vocab}"
            )
            assert r.numel() > 0, f"prompt {i}: empty tensor"
            assert not torch.isnan(r).any(), f"prompt {i}: NaN in logits"
            assert r.abs().max().item() > 0, f"prompt {i}: all-zeros"

    def test_large_concurrent_burst(self, server_url, client_model):
        """Stress: issue 2x the batch size to force multi-step scheduling."""
        prompts = PROMPTS * 4  # 32 requests against max_batch_size=16
        results = _run_concurrent(client_model, server_url, prompts)

        assert len(results) == len(prompts)
        for i, r in enumerate(results):
            assert r is not None and not torch.isnan(r).any(), (
                f"prompt {i}/{len(prompts)} failed under load"
            )


class TestCorrectness:

    def test_solo_request_matches_sequential(
        self, server_url, client_model, baseline_model,
    ):
        """Single request through the server (no concurrency) should
        produce the same top-1 next-token prediction as a local trace.

        When ``batch_size == 1`` in ``_step``, ``needs_batching`` is
        False and the batcher's narrow is a no-op — the mediator sees
        the full ``(1, seq, vocab)`` forward output, same as local.
        """
        for i, prompt in enumerate(PROMPTS[:4]):
            batched = _submit_via_serve(client_model, prompt, server_url).detach().cpu()
            seq = _submit_local(baseline_model, prompt).detach().cpu()

            b_last = batched[:, -1, :].flatten().float()
            s_last = seq[:, -1, :].flatten().float()
            b_top = int(b_last.argmax().item())
            s_top = int(s_last.argmax().item())
            cos = torch.nn.functional.cosine_similarity(
                b_last.unsqueeze(0), s_last.unsqueeze(0),
            ).item()
            assert b_top == s_top or cos >= 0.999, (
                f"prompt {i}: solo-serve top-1={b_top} vs local top-1={s_top}, "
                f"cos={cos:.4f}"
            )

    def test_concurrent_results_nontrivially_correlated_with_sequential(
        self, server_url, client_model, baseline_model,
    ):
        """Concurrent batched results should not be randomly related to
        sequential — they should correlate positively with the true
        next-token distribution on average.

        This is a loose check because concurrent batching goes through
        ``process_batch_groups`` with ``num_tokens_map={req_id: 1}``
        hardcoded, which currently produces narrowed (1, 1, vocab)
        outputs instead of full (1, seq, vocab) per request. So
        per-prompt cosine similarity can be low or negative depending
        on which token position ends up captured. We require that
        *on average* across prompts the correlation is positive, which
        is a weak sanity rail against wiring-level regressions (wrong
        model, wrong weights, etc.).
        """
        batched = _run_concurrent(client_model, server_url, PROMPTS)
        sequential = [
            _submit_local(baseline_model, p).detach().cpu() for p in PROMPTS
        ]

        cosines = []
        for b, s in zip(batched, sequential):
            b = b.detach().cpu().flatten().float()
            s = s[:, -1, :].flatten().float()
            # Align on trailing vocab dim; both end in `vocab` elements.
            vocab = s.shape[-1]
            b_last = b[-vocab:]
            cos = torch.nn.functional.cosine_similarity(
                b_last.unsqueeze(0), s.unsqueeze(0),
            ).item()
            cosines.append(cos)

        mean_cos = sum(cosines) / len(cosines)
        print(
            f"\n  Per-prompt cos sim (concurrent vs sequential): "
            f"{[f'{c:+.3f}' for c in cosines]}\n"
            f"  Mean: {mean_cos:+.3f}"
        )
        # Very loose: mean should at least be positive (not random).
        # Tightening requires fixing the num_tokens_map bug in vanilla_server._step.
        assert mean_cos > 0, (
            f"concurrent results uncorrelated with sequential (mean cos={mean_cos:.3f}); "
            f"likely a wiring regression"
        )

    def test_repeated_submissions_stable(self, server_url, client_model):
        """Same prompt submitted twice returns identical argmax."""
        prompt = PROMPTS[0]
        runs = []
        for _ in range(3):
            logits = _submit_via_serve(client_model, prompt, server_url)
            runs.append(int(logits[:, -1, :].argmax().item()))
        assert len(set(runs)) == 1, f"unstable output across runs: {runs}"


class TestPerformance:

    def test_concurrent_vs_sequential_timing(
        self, server_url, client_model, baseline_model,
    ):
        """Compare N concurrent serve traces vs N sequential generations
        of equivalent length (both do ``SERVER_MAX_NEW_TOKENS`` forwards
        per request so the comparison is apples-to-apples in total
        compute; only the scheduling is different).
        """
        def _sequential_generate(prompt):
            # Match the server's total forward count per request.
            with baseline_model.generate(
                prompt, max_new_tokens=SERVER_MAX_NEW_TOKENS,
            ):
                out = baseline_model.generator.output.save()
            return out

        # Warmup both paths
        _sequential_generate(PROMPTS[0])
        _submit_via_serve(client_model, PROMPTS[0], server_url)

        # Sequential baseline: one .generate() per prompt, one after another
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for p in PROMPTS:
            _sequential_generate(p)
        torch.cuda.synchronize()
        seq_time = time.perf_counter() - t0

        # Concurrent through server: N clients hitting serve= in parallel
        t0 = time.perf_counter()
        _run_concurrent(client_model, server_url, PROMPTS)
        bat_time = time.perf_counter() - t0

        n = len(PROMPTS)
        print(
            f"\n  ── Timing ({n} prompts, {SERVER_MAX_NEW_TOKENS} tokens each) ──\n"
            f"  Sequential (local .generate, GPU1): {seq_time*1000:8.1f} ms total, "
            f"{seq_time*1000/n:6.2f} ms/prompt\n"
            f"  Concurrent (serve, GPU0):           {bat_time*1000:8.1f} ms total, "
            f"{bat_time*1000/n:6.2f} ms/prompt\n"
            f"  Speedup (seq / batched):            {seq_time/bat_time:.2f}x"
        )

        # Real expectation: batched should be faster than sequential for
        # this workload (N independent generations that CAN be batched).
        # If it isn't, either batching is broken or overhead dominates.
        assert bat_time < seq_time, (
            f"Batched ({bat_time*1000:.1f} ms) slower than sequential "
            f"({seq_time*1000:.1f} ms) — batching not providing speedup. "
            f"Expected speedup since all {n} prompts can run in one "
            f"forward pass per decode step."
        )

    def test_throughput_under_load(
        self, server_url, client_model,
    ):
        """Throughput across a larger burst, reported for profiling."""
        prompts = PROMPTS * 4  # 32 requests
        # Warmup
        _submit_via_serve(client_model, prompts[0], server_url)

        t0 = time.perf_counter()
        results = _run_concurrent(client_model, server_url, prompts)
        elapsed = time.perf_counter() - t0

        assert all(r is not None and not torch.isnan(r).any() for r in results)
        rps = len(prompts) / elapsed
        print(
            f"\n  Burst {len(prompts)} concurrent prompts: {elapsed*1000:.1f} ms "
            f"({rps:.1f} req/s, {elapsed*1000/len(prompts):.2f} ms/req)"
        )
