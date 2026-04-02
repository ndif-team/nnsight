"""Stress test: concurrent requests to nnsight-serve with PP=2.

Starts a server with PP=2, sends N concurrent requests with various
intervention patterns, measures throughput and verifies correctness.

Run: CUDA_VISIBLE_DEVICES=0,1 python tests/stress_pp_serve.py
Requires 2 GPUs, uvicorn, httpx.
"""

import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 6688
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
PROMPT = "The Eiffel Tower is located in the city of"


def wait_for_server(timeout=120):
    """Wait until /health responds."""
    import httpx
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = httpx.get(f"{SERVER_URL}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def run_trace(model, intervention_name):
    """Run a single trace and return (name, saves_dict_or_error, elapsed_ms)."""
    t0 = time.perf_counter()
    try:
        if intervention_name == "logits":
            with model.trace(PROMPT, temperature=0.0, top_p=1, serve=SERVER_URL):
                logits = model.logits.output.save()
            return ("logits", {"logits": logits}, (time.perf_counter() - t0) * 1000)

        elif intervention_name == "hidden_l0":
            with model.trace(PROMPT, temperature=0.0, top_p=1, serve=SERVER_URL):
                h0 = model.transformer.h[0].output[0].save()
            return ("hidden_l0", {"h0": h0}, (time.perf_counter() - t0) * 1000)

        elif intervention_name == "hidden_l11":
            with model.trace(PROMPT, temperature=0.0, top_p=1, serve=SERVER_URL):
                h11 = model.transformer.h[11].output[0].save()
            return ("hidden_l11", {"h11": h11}, (time.perf_counter() - t0) * 1000)

        elif intervention_name == "cross_read":
            with model.trace(PROMPT, temperature=0.0, top_p=1, serve=SERVER_URL):
                h0 = model.transformer.h[0].output[0].save()
                logits = model.logits.output.save()
            return ("cross_read", {"h0": h0, "logits": logits}, (time.perf_counter() - t0) * 1000)

        elif intervention_name == "multigen":
            with model.trace(PROMPT, temperature=0.0, top_p=1, max_tokens=3, serve=SERVER_URL) as tracer:
                logit_list = list().save()
                for step in tracer.iter[0:3]:
                    logit_list.append(model.logits.output)
            return ("multigen", {"logit_list": logit_list}, (time.perf_counter() - t0) * 1000)

        else:
            return (intervention_name, {"error": f"Unknown intervention"}, 0)

    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return (intervention_name, {"error": str(e)[:200]}, elapsed)


def run_stress_test():
    from nnsight.modeling.vllm import VLLM

    # Meta model — no GPU needed
    model = VLLM("openai-community/gpt2")

    print("Waiting for server...")
    if not wait_for_server():
        print("ERROR: Server didn't start in time")
        return False

    print("Server is ready.\n")
    failures = []

    # --- Test 1: Single request ---
    print("Test 1: single logits request")
    name, saves, ms = run_trace(model, "logits")
    if "error" in saves:
        print(f"  FAIL: {saves['error']}")
        failures.append("test1")
    else:
        print(f"  OK: {saves['logits'].shape} in {ms:.0f}ms")

    # --- Test 2: 5 concurrent logits ---
    print("Test 2: 5 concurrent logits")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(run_trace, model, "logits") for _ in range(5)]
        results = [f.result() for f in futures]
    t1 = time.perf_counter()
    ok = sum(1 for _, s, _ in results if "error" not in s)
    times = [ms for _, s, ms in results if "error" not in s]
    print(f"  {ok}/5 OK, wall={t1-t0:.2f}s, mean={sum(times)/len(times):.0f}ms" if times else f"  {ok}/5 OK")
    if ok < 5:
        for n, s, _ in results:
            if "error" in s:
                print(f"    FAIL: {s['error'][:100]}")
        failures.append("test2")

    # --- Test 3: 5 concurrent hidden (stage 0) ---
    print("Test 3: 5 concurrent hidden layer 0 (stage 0)")
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(run_trace, model, "hidden_l0") for _ in range(5)]
        results = [f.result() for f in futures]
    ok = sum(1 for _, s, _ in results if "error" not in s)
    print(f"  {ok}/5 OK")
    if ok < 5:
        failures.append("test3")

    # --- Test 4: 10 concurrent mixed interventions ---
    print("Test 4: 10 concurrent mixed (logits + hidden + cross_read)")
    interventions = ["logits", "hidden_l0", "hidden_l11", "cross_read", "logits",
                     "hidden_l0", "hidden_l11", "cross_read", "logits", "logits"]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(run_trace, model, iv) for iv in interventions]
        results = [f.result() for f in futures]
    t1 = time.perf_counter()
    ok = sum(1 for _, s, _ in results if "error" not in s)
    print(f"  {ok}/10 OK, wall={t1-t0:.2f}s ({10/(t1-t0):.1f} req/s)")
    if ok < 10:
        for n, s, _ in results:
            if "error" in s:
                print(f"    {n}: {s['error'][:100]}")
        failures.append("test4")

    # --- Test 5: 5 concurrent multi-token generation ---
    print("Test 5: 5 concurrent multi-token generation (3 tokens each)")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(run_trace, model, "multigen") for _ in range(5)]
        results = [f.result() for f in futures]
    t1 = time.perf_counter()
    ok = sum(1 for _, s, _ in results if "error" not in s)
    print(f"  {ok}/5 OK, wall={t1-t0:.2f}s")
    if ok < 5:
        for n, s, _ in results:
            if "error" in s:
                print(f"    {n}: {s['error'][:100]}")
        failures.append("test5")

    # --- Test 6: Sustained load (20 sequential requests) ---
    print("Test 6: 20 sequential logits requests")
    t0 = time.perf_counter()
    ok = 0
    for i in range(20):
        _, saves, _ = run_trace(model, "logits")
        if "error" not in saves:
            ok += 1
    t1 = time.perf_counter()
    print(f"  {ok}/20 OK, total={t1-t0:.2f}s ({20/(t1-t0):.1f} req/s)")
    if ok < 20:
        failures.append("test6")

    # --- Summary ---
    print()
    if failures:
        print(f"FAILED ({len(failures)}): {failures}")
        return False
    else:
        print("ALL PASS")
        return True


if __name__ == "__main__":
    env = os.environ.copy()
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = "0,1"

    server_cmd = [
        sys.executable, "-m", "nnsight.modeling.vllm.serve.cli",
        "openai-community/gpt2",
        "--port", str(SERVER_PORT),
        "--pipeline-parallel-size", "2",
        "--gpu-memory-utilization", "0.1",
    ]

    print(f"Starting server: {' '.join(server_cmd[-6:])}")
    server_proc = subprocess.Popen(
        server_cmd, env=env,
        stdout=sys.stderr, stderr=sys.stderr,
    )

    try:
        success = run_stress_test()
    except KeyboardInterrupt:
        print("\nInterrupted")
        success = False
    finally:
        server_proc.send_signal(signal.SIGINT)
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("Server stopped.")

    sys.exit(0 if success else 1)
