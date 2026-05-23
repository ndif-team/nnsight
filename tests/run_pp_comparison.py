"""PP=1 vs PP=2 comparison tests for GPT-2.

Run: CUDA_VISIBLE_DEVICES=6,7 conda run -n ndif-dev python tests/run_pp_comparison.py
"""
import torch
import os
import sys

def cosine(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def run_pp(pp_size):
    """Load model and run all scenarios, return results dict."""
    from nnsight.modeling.vllm import VLLM

    model = VLLM(
        'gpt2',
        pipeline_parallel_size=pp_size,
        gpu_memory_utilization=0.1,
        dispatch=True,
    )

    results = {}

    # A: Logits
    with model.trace('The Eiffel Tower is in', temperature=0.0, top_p=1):
        logits = model.logits.output.save()
    results['logits'] = logits.detach().cpu().float()
    results['argmax'] = int(logits.argmax(dim=-1).item())

    # B: Layer 0 hidden state
    with model.trace('Hello world', temperature=0.0, top_p=1):
        h0 = model.transformer.h[0].output[0].save()
    results['h0'] = h0.detach().cpu().float()

    # C: Layer 11 hidden state
    with model.trace('Hello world', temperature=0.0, top_p=1):
        h11 = model.transformer.h[11].output[0].save()
    results['h11'] = h11.detach().cpu().float()

    # D: Both stages in one trace
    with model.trace('Hello world', temperature=0.0, top_p=1):
        h0_d = model.transformer.h[0].output[0].save()
        h11_d = model.transformer.h[11].output[0].save()
    results['h0_d'] = h0_d.detach().cpu().float()
    results['h11_d'] = h11_d.detach().cpu().float()

    return results


def main():
    pp = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    if pp in (1, 2):
        # Worker mode: run one PP config and save results
        results = run_pp(pp)
        torch.save(results, f'/tmp/pp{pp}_results.pt')
        print(f'PP={pp} done. Saved to /tmp/pp{pp}_results.pt')
        return

    # Orchestrator mode: run both and compare
    import subprocess

    gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '6,7')
    gpu_list = gpus.split(',')

    print("=" * 60)
    print("PP=1 vs PP=2 Comparison (GPT-2)")
    print("=" * 60)

    for pp_size in [1, 2]:
        visible = gpu_list[0] if pp_size == 1 else gpus
        port = 29710 + pp_size
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = visible
        env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        env['MASTER_PORT'] = str(port)

        print(f"\nRunning PP={pp_size} on GPU(s) {visible}...")
        r = subprocess.run(
            ['conda', 'run', '-n', 'ndif-dev', 'python', __file__, str(pp_size)],
            env=env, capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            print(f"  FAILED!")
            for line in r.stderr.split('\n'):
                if 'Error' in line:
                    print(f"    {line}")
            return

    r1 = torch.load('/tmp/pp1_results.pt', weights_only=False)
    r2 = torch.load('/tmp/pp2_results.pt', weights_only=False)

    print("\n--- Results ---")
    print(f"A: Logits argmax  PP=1={r1['argmax']}  PP=2={r2['argmax']}  same={r1['argmax']==r2['argmax']}")
    print(f"   Cosine similarity: {cosine(r1['logits'], r2['logits']):.6f}")

    print(f"B: Layer 0 cosine: {cosine(r1['h0'], r2['h0']):.6f}")
    print(f"C: Layer 11 cosine: {cosine(r1['h11'], r2['h11']):.6f}")
    print(f"D: Both stages  h0={cosine(r1['h0_d'], r2['h0_d']):.6f}  h11={cosine(r1['h11_d'], r2['h11_d']):.6f}")

    all_pass = all([
        cosine(r1['logits'], r2['logits']) > 0.90,
        cosine(r1['h0'], r2['h0']) > 0.99,
        cosine(r1['h11'], r2['h11']) > 0.90,
    ])
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")


if __name__ == '__main__':
    main()
