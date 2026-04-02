"""End-to-end PP tests including cross-stage interventions.

Tests:
  1-5: In-stage operations (no cross-rank pull needed)
  6-7: PP+TP combined
  8: Cross-stage READ (layer 0 from stage 1) — exercises LazyRemoteTensor + pull
  9: Cross-stage WRITE (layer 2 → layer 8) — exercises cross-stage data flow
 10: Cross-stage multi-token gen — pull per generation step
 11: Cross-stage read with PP=2 vs PP=1 comparison

Requires at least 4 GPUs.
Run: CUDA_VISIBLE_DEVICES=0,1,2,3 python tests/test_pp_pull.py
"""

import json
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pp_pull_worker.py")


def run(scenario, gpus, pp, tp=1, **kwargs):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = f.name
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus
        cmd = [sys.executable, WORKER, scenario, "--pp", str(pp), "--tp", str(tp), "--output", out]
        for k, v in kwargs.items():
            cmd.extend([f"--{k}", str(v)])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env, cwd=REPO_ROOT)
        if r.returncode != 0:
            # Extract useful error from stderr
            err_lines = [l for l in r.stderr.split('\n') if 'Error' in l or 'error' in l.lower()]
            return {"status": "error", "error": '\n'.join(err_lines[-5:]) if err_lines else r.stderr[-500:]}
        with open(out) as f:
            return json.load(f)
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass


if __name__ == '__main__':
    failures = []

    def test(name, result, check=None):
        if result.get("status") != "ok":
            err = result.get("error", "unknown")[:300]
            tb = result.get("traceback", "")[-500:]
            print(f"  FAIL: {err}")
            if tb:
                # Print last meaningful line of traceback
                for line in reversed(tb.strip().split('\n')):
                    if line.strip() and not line.startswith(' '):
                        print(f"        {line.strip()}")
                        break
            failures.append(f"{name}: {err[:150]}")
            return
        if check:
            ok, msg = check(result)
            if not ok:
                print(f"  FAIL: {msg}")
                failures.append(f"{name}: {msg}")
                return
        print(f"  OK")

    PP2 = "0,1"
    PP2_TP2 = "0,1,2,3"

    # ---- In-stage tests ----
    print("Test 1: basic trace PP=2")
    test("basic_pp2", run("basic_trace", PP2, pp=2),
         lambda r: (r["top_token"] == " Paris", f"got {r['top_token']!r}"))

    print("Test 2: hidden layer 0 (stage 0)")
    test("hidden_l0", run("hidden", PP2, pp=2, layer=0))

    print("Test 3: hidden layer 11 (stage 1)")
    test("hidden_l11", run("hidden", PP2, pp=2, layer=11))

    print("Test 4: logits PP=2")
    test("logits_pp2", run("logits", PP2, pp=2),
         lambda r: (r["top_token"] == " Paris", f"got {r['top_token']!r}"))

    print("Test 5: multi-token PP=2 (3 tokens)")
    test("multigen_pp2", run("multigen", PP2, pp=2, max_tokens=3),
         lambda r: (r["num_steps"] == 3, f"got {r['num_steps']} steps"))

    # ---- PP+TP combined ----
    print("Test 6: basic trace PP=2 TP=2")
    test("basic_pp2tp2", run("basic_trace", PP2_TP2, pp=2, tp=2),
         lambda r: (r["top_token"] == " Paris", f"got {r['top_token']!r}"))

    print("Test 7: multi-token PP=2 TP=2 (3 tokens)")
    test("multigen_pp2tp2", run("multigen", PP2_TP2, pp=2, tp=2, max_tokens=3),
         lambda r: (r["num_steps"] == 3, f"got {r['num_steps']} steps"))

    # ---- Cross-stage tests (end-to-end pull) ----
    print("Test 8: cross-stage READ (layer 0 from stage 1)")
    test("cross_read", run("cross_stage_read", PP2, pp=2),
         lambda r: (r["top_token"] == " Paris" and len(r["h0_shape"]) > 0,
                     f"got {r.get('top_token')!r}, h0_shape={r.get('h0_shape')}"))

    print("Test 9: cross-stage WRITE (layer 2 → layer 8)")
    test("cross_write", run("cross_stage_write", PP2, pp=2))

    print("Test 10: cross-stage multi-token (3 steps)")
    test("cross_multigen", run("cross_stage_multigen", PP2, pp=2, max_tokens=3),
         lambda r: (r["num_steps"] == 3 and len(r["h0_shapes"]) == 3,
                     f"steps={r.get('num_steps')}, h0_shapes={r.get('h0_shapes')}"))

    # ---- Cross-stage PP=1 vs PP=2 comparison ----
    print("Test 11: cross-stage read PP=1 vs PP=2 comparison")
    pp1 = run("cross_stage_read", "0", pp=1)
    pp2 = run("cross_stage_read", PP2, pp=2)
    if pp1.get("status") == "ok" and pp2.get("status") == "ok":
        h0_diff = abs(pp1["h0_mean"] - pp2["h0_mean"])
        same_token = pp1["argmax"] == pp2["argmax"]
        if same_token and h0_diff < 1.0:
            print(f"  OK: same token={pp1['top_token']!r}, h0_mean_diff={h0_diff:.4f}")
        else:
            msg = f"token PP1={pp1['top_token']!r} PP2={pp2['top_token']!r}, h0_mean_diff={h0_diff:.4f}"
            print(f"  FAIL: {msg}")
            failures.append(f"cross_compare: {msg}")
    else:
        err = pp1.get("error", "") or pp2.get("error", "")
        print(f"  FAIL: {err[:200]}")
        failures.append(f"cross_compare: {err[:150]}")

    # ---- Complex intervention tests ----
    print("Test 12: save ALL 12 layers (spans both stages)")
    test("all_layers", run("save_all_layers", PP2, pp=2),
         lambda r: (r["num_layers"] == 12 and r["top_token"] == " Paris",
                     f"layers={r.get('num_layers')}, token={r.get('top_token')!r}"))

    print("Test 13: cross-stage clone + modify (h2*0.5 → h8)")
    test("clone_modify", run("cross_clone_modify", PP2, pp=2))

    print("Test 14: ablation (zero layer 3, zero layer 8)")
    r14 = run("ablation", PP2, pp=2)
    if r14.get("status") == "ok":
        if r14.get("l3_changed") and r14.get("l8_changed"):
            print(f"  OK: baseline={r14['baseline']!r}, ablated_l3={r14['ablated_l3']!r}, ablated_l8={r14['ablated_l8']!r}")
        else:
            msg = f"ablation had no effect: l3_changed={r14.get('l3_changed')}, l8_changed={r14.get('l8_changed')}"
            print(f"  FAIL: {msg}")
            failures.append(f"ablation: {msg}")
    else:
        print(f"  FAIL: {r14.get('error', '')[:200]}")
        failures.append(f"ablation: {r14.get('error', '')[:150]}")

    print("Test 15: steering (h2 mean → add to h8)")
    test("steering", run("steering", PP2, pp=2))

    print("Test 16: PP=1 vs PP=2 multi-layer comparison (h0, h5, h6, h11)")
    pp1 = run("cross_compare", "0", pp=1)
    pp2 = run("cross_compare", PP2, pp=2)
    if pp1.get("status") == "ok" and pp2.get("status") == "ok":
        diffs = {k: abs(pp1[k] - pp2[k]) for k in ["h0_mean", "h5_mean", "h6_mean", "h11_mean"]}
        same_token = pp1["argmax"] == pp2["argmax"]
        max_diff = max(diffs.values())
        if same_token and max_diff < 1.0:
            print(f"  OK: same token={pp1['top_token']!r}, max_diff={max_diff:.4f}")
        else:
            print(f"  FAIL: token PP1={pp1['top_token']!r} PP2={pp2['top_token']!r}, diffs={diffs}")
            failures.append("multi_layer_compare")
    else:
        err = pp1.get("error", "") or pp2.get("error", "")
        print(f"  FAIL: {err[:200]}")
        failures.append(f"multi_layer_compare: {err[:150]}")

    print("Test 17: multi-token with cross-stage write per step")
    test("multigen_cross_write", run("multigen_cross_write", PP2, pp=2, max_tokens=3),
         lambda r: (r["num_steps"] == 3, f"steps={r.get('num_steps')}"))

    # ---- PP=2 TP=2 complex tests ----
    print("Test 18: save all layers PP=2 TP=2")
    test("all_layers_tp2", run("save_all_layers", PP2_TP2, pp=2, tp=2),
         lambda r: (r["num_layers"] == 12, f"layers={r.get('num_layers')}"))

    print("Test 19: ablation PP=2 TP=2")
    r19 = run("ablation", PP2_TP2, pp=2, tp=2)
    if r19.get("status") == "ok":
        if r19.get("l3_changed") and r19.get("l8_changed"):
            print(f"  OK")
        else:
            print(f"  FAIL: l3_changed={r19.get('l3_changed')}, l8_changed={r19.get('l8_changed')}")
            failures.append("ablation_tp2")
    else:
        print(f"  FAIL: {r19.get('error', '')[:200]}")
        failures.append(f"ablation_tp2: {r19.get('error', '')[:150]}")

    # ---- Summary ----
    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL PASS")
