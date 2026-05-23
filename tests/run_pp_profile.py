"""Direct PP overhead profiling. Loads models once, runs all scenarios.

Run: CUDA_VISIBLE_DEVICES=6,7 conda run -n ndif-dev python tests/run_pp_profile.py
"""
import time
import statistics
import torch
import os
import sys

def timed(fn, n_warmup=2, n_runs=10):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times

def run_scenarios(model, label):
    results = {}
    prompt = "The Eiffel Tower is located in the city of"

    # A: Baseline
    def baseline():
        with model.trace(prompt, temperature=0.0, top_p=1):
            pass
    results['A: baseline'] = timed(baseline)

    # B: Save logits
    def save_logits():
        with model.trace(prompt, temperature=0.0, top_p=1):
            _ = model.logits.output.save()
    results['B: save_logits'] = timed(save_logits)

    # C: Save early layer
    def save_early():
        with model.trace(prompt, temperature=0.0, top_p=1):
            _ = model.transformer.h[0].output[0].save()
    results['C: save_h0'] = timed(save_early)

    # D: Save both stages
    def save_both():
        with model.trace(prompt, temperature=0.0, top_p=1):
            _ = model.transformer.h[0].output[0].save()
            _ = model.transformer.h[11].output[0].save()
    results['D: save_both'] = timed(save_both)

    # E: Cross-stage write
    def cross_write():
        with model.trace(prompt, temperature=0.0, top_p=1):
            h = model.transformer.h[2].output[0]
            model.transformer.h[8].output[0][:] = h
    results['E: cross_write'] = timed(cross_write)

    # J: Long-distance (wte -> h[11])
    def long_distance():
        with model.trace(prompt, temperature=0.0, top_p=1):
            emb = model.transformer.wte.output.save()
            _ = model.transformer.h[11].output[0].save()
    results['J: long_dist'] = timed(long_distance)

    # K: Two traces back-to-back
    def two_traces():
        with model.trace("Hello", temperature=0.0, top_p=1):
            _ = model.logits.output.save()
        with model.trace("World", temperature=0.0, top_p=1):
            _ = model.logits.output.save()
    results['K: two_traces'] = timed(two_traces)

    # L: Save all 12 layers
    def save_all():
        with model.trace(prompt, temperature=0.0, top_p=1):
            for i in range(12):
                _ = model.transformer.h[i].output[0].save()
    results['L: save_all_12'] = timed(save_all)

    return results

def main():
    from nnsight.modeling.vllm import VLLM

    pp = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    if pp in (1, 2):
        print(f"Loading GPT-2 PP={pp}...", flush=True)
        kwargs = {"gpu_memory_utilization": 0.1, "dispatch": True}
        if pp > 1:
            kwargs["pipeline_parallel_size"] = pp
        model = VLLM("openai-community/gpt2", **kwargs)

        print(f"Running scenarios PP={pp}...", flush=True)
        results = run_scenarios(model, f"PP={pp}")
        torch.save(results, f"/tmp/pp{pp}_profile.pt")
        print(f"PP={pp} done.", flush=True)
        return

    # Orchestrator
    import subprocess

    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "6,7")
    gpu_list = gpus.split(",")

    print("=" * 70)
    print("  PP Overhead Profiling (GPT-2, 10 runs)")
    print("=" * 70)

    for pp_size in [1, 2]:
        visible = gpu_list[0] if pp_size == 1 else gpus
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env["MASTER_PORT"] = str(29720 + pp_size)

        print(f"\nRunning PP={pp_size} on GPU(s) {visible}...")
        r = subprocess.run(
            ["conda", "run", "-n", "ndif-dev", "python", __file__, str(pp_size)],
            env=env, capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            print(f"  FAILED!")
            for line in r.stderr.split('\n'):
                if 'Error' in line:
                    print(f"    {line}")
            return

    r1 = torch.load("/tmp/pp1_profile.pt", weights_only=False)
    r2 = torch.load("/tmp/pp2_profile.pt", weights_only=False)

    print(f"\n{'Scenario':<30} {'PP=1 (ms)':>12} {'PP=2 (ms)':>12} {'Overhead':>10}")
    print("-" * 70)

    for key in r1:
        t1 = r1[key]
        t2 = r2.get(key, [])
        m1 = statistics.mean(t1) * 1000
        m2 = statistics.mean(t2) * 1000 if t2 else float('nan')
        s1 = statistics.stdev(t1) * 1000 if len(t1) > 1 else 0
        s2 = statistics.stdev(t2) * 1000 if len(t2) > 1 else 0
        overhead = ((m2 - m1) / m1 * 100) if m1 > 0 and t2 else float('nan')
        print(f"{key:<30} {m1:8.1f}±{s1:4.1f} {m2:8.1f}±{s2:4.1f} {overhead:>+8.1f}%")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
