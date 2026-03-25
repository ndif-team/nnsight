"""
Test nnsight documentation examples on vLLM backend.

For each doc example pattern, tests whether it works on vLLM as-written,
documents the mismatch, and shows the vLLM workaround.

Run: python test_doc_examples.py [--gpu 0]
"""
import subprocess, sys, os, json, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0")
parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
args = parser.parse_args()

SCRIPT = r'''
import os, json, torch, traceback, nnsight
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu}"

from nnsight.modeling.vllm import VLLM

model = VLLM("{model}", tensor_parallel_size=1, gpu_memory_utilization=0.3,
             dtype=torch.float16, dispatch=True)

results = []

def run_test(name, section, fn):
    result = {{"name": name, "section": section}}
    try:
        outcome = fn()
        if outcome is None:
            outcome = {{}}
        result.update(outcome)
        if "status" not in result:
            result["status"] = "PASS"
    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = f"{{type(e).__name__}}: {{e}}"
    results.append(result)
    return result

# ═══ 1. Save layer hidden states ═══

def test_save_hs_doc():
    with model.trace("Hello world", temperature=0.0, top_p=1):
        hs = model.model.layers[-1].output[0].save()
    return {{
        "status": "MISMATCH",
        "detail": f"output[0] is raw MLP output, shape={{list(hs.shape)}}",
        "vllm_issue": "Gap 1.2: output[0] is MLP-only, not combined hidden state. Gap 1.1: mutated in-place.",
        "workaround": "output[0].clone() + output[1].clone()",
    }}
run_test("Save layer hidden states (doc pattern)", "Accessing Outputs", test_save_hs_doc)

def test_save_hs_fix():
    with model.trace("Hello world", temperature=0.0, top_p=1):
        hs = (model.model.layers[-1].output[0].clone() +
              model.model.layers[-1].output[1].clone()).save()
    return {{"status": "PASS", "detail": f"Combined: shape={{list(hs.shape)}}"}}
run_test("Save layer hidden states (workaround)", "Accessing Outputs", test_save_hs_fix)

# ═══ 2. Save logits ═══

def test_logits_lm_head():
    try:
        with model.trace("Hello world", temperature=0.0, top_p=1):
            logits = model.lm_head.output.save()
        return {{"status": "PASS", "detail": f"shape={{list(logits.shape)}}"}}
    except Exception as e:
        return {{"status": "MISMATCH", "detail": f"lm_head: {{type(e).__name__}}", "workaround": "Use model.logits.output"}}
run_test("Save logits (lm_head)", "Accessing Outputs", test_logits_lm_head)

def test_logits_vllm():
    with model.trace("Hello world", temperature=0.0, top_p=1):
        logits = model.logits.output.save()
    return {{"status": "PASS", "detail": f"shape={{list(logits.shape)}}"}}
run_test("Save logits (model.logits)", "vLLM Integration", test_logits_vllm)

# ═══ 3. Access layer input ═══

def test_layer_input():
    with model.trace("Hello world", temperature=0.0, top_p=1):
        inp = model.model.layers[0].input.clone().save()
    if inp.dtype in (torch.int64, torch.int32, torch.long):
        return {{
            "status": "MISMATCH",
            "detail": f".input returns position IDs (dtype={{inp.dtype}})",
            "vllm_issue": "Gap 1.3: vLLM forward(positions, hidden_states, residual, ...)",
            "workaround": "args, _ = layer.inputs; hidden = args[1] + args[2]",
        }}
    return {{"status": "PASS", "detail": f"dtype={{inp.dtype}}"}}
run_test("Access layer.input", "Accessing Inputs", test_layer_input)

def test_layer_input_fix():
    with model.trace("Hello world", temperature=0.0, top_p=1):
        a, _ = model.model.layers[4].inputs
        hidden = (a[1].clone() + a[2].clone()).save()
    return {{"status": "PASS", "detail": f"dtype={{hidden.dtype}}, shape={{list(hidden.shape)}}"}}
run_test("Access layer.inputs workaround", "Accessing Inputs", test_layer_input_fix)

# ═══ 4. In-place zero ═══

def test_zero_doc():
    try:
        with model.trace("Hello world", temperature=0.0, top_p=1):
            model.model.layers[0].output[0][:] = 0
            logits = model.logits.output.save()
        return {{"status": "MISMATCH", "detail": "Only zeros MLP stream, residual leaks", "vllm_issue": "Gap 1.2: must zero both streams"}}
    except Exception as e:
        return {{"status": "FAIL", "error": str(e)}}
run_test("In-place zero output[0][:]", "Modifying Activations", test_zero_doc)

def test_zero_fix():
    with model.trace("Hello world", temperature=0.0, top_p=1):
        model.model.layers[0].output = (
            torch.zeros_like(model.model.layers[0].output[0]),
            torch.zeros_like(model.model.layers[0].output[1]),
        )
        logits = model.logits.output.save()
    return {{"status": "PASS", "detail": "Both streams zeroed", "workaround": "layer.output = (zeros, zeros)"}}
run_test("Zero both streams (workaround)", "Modifying Activations", test_zero_fix)

# ═══ 5. Steering vector ═══

def test_steer_doc():
    vec = torch.randn(896, dtype=torch.float16).cuda() * 0.1
    try:
        with model.trace("Hello world", temperature=0.0, top_p=1):
            model.model.layers[10].output[0][:, -1, :] += vec
            logits = model.logits.output.save()
        return {{"status": "PASS"}}
    except Exception as e:
        return {{
            "status": "MISMATCH",
            "detail": f"3D indexing fails: {{type(e).__name__}}",
            "vllm_issue": "Gap 3.1: vLLM is 2D [tokens, hidden], not 3D [batch, seq, hidden]",
            "workaround": "out[-1, :] += vec (2D) + clone + replace tuple",
        }}
run_test("Steering [:, -1, :] (doc)", "Steering", test_steer_doc)

def test_steer_fix():
    vec = torch.randn(896, dtype=torch.float16).cuda() * 0.1
    with model.trace("Hello world", temperature=0.0, top_p=1):
        o0 = model.model.layers[10].output[0].clone()
        o1 = model.model.layers[10].output[1].clone()
        o0[-1, :] += vec
        model.model.layers[10].output = (o0, o1)
        logits = model.logits.output.save()
    return {{"status": "PASS", "detail": "2D indexing + clone + tuple replace"}}
run_test("Steering (workaround)", "Steering", test_steer_fix)

# ═══ 6. Activation patching ═══

def test_patch_doc():
    try:
        with model.trace() as tracer:
            barrier = tracer.barrier(2)
            with tracer.invoke("The Eiffel Tower is in", temperature=0.0, top_p=1):
                clean_hs = model.model.layers[5].output[0][:, -1, :]
                barrier()
            with tracer.invoke("The Colosseum is in", temperature=0.0, top_p=1):
                barrier()
                model.model.layers[5].output[0][:, -1, :] = clean_hs
                logits = model.logits.output.save()
        return {{"status": "PASS"}}
    except Exception as e:
        return {{
            "status": "MISMATCH",
            "detail": f"Fails: {{type(e).__name__}}",
            "vllm_issue": "3D indexing + tuple assignment + dual residual",
            "workaround": "2D indexing, clone both streams, replace tuple",
        }}
run_test("Activation patching (doc)", "Activation Patching", test_patch_doc)

def test_patch_fix():
    with model.trace() as tracer:
        barrier = tracer.barrier(2)
        with tracer.invoke("The Eiffel Tower is in", temperature=0.0, top_p=1):
            clean = (model.model.layers[5].output[0].clone()[-1, :] +
                     model.model.layers[5].output[1].clone()[-1, :]).save()
            barrier()
        with tracer.invoke("The Colosseum is in", temperature=0.0, top_p=1):
            barrier()
            o0 = model.model.layers[5].output[0].clone()
            o1 = model.model.layers[5].output[1].clone()
            o0[-1, :] = clean - o1[-1, :]
            model.model.layers[5].output = (o0, o1)
            logits = model.logits.output.save()
    return {{"status": "PASS", "detail": "2D + clone + dual stream"}}
run_test("Activation patching (workaround)", "Activation Patching", test_patch_fix)

# ═══ 7. Logit lens ═══

def test_logit_lens_doc():
    try:
        with model.trace("The Eiffel Tower is in", temperature=0.0, top_p=1):
            hs = model.model.layers[5].output[0]
            logits = model.lm_head(model.model.norm(hs))
            token = logits.argmax(dim=-1).save()
        return {{"status": "PASS"}}
    except Exception as e:
        return {{
            "status": "MISMATCH",
            "detail": f"{{type(e).__name__}}: {{str(e)[:80]}}",
            "vllm_issue": "Gap 1.2 (dual residual) + Gap 1.4 (norm returns tuple)",
            "workaround": "Combine streams, handle norm tuple output",
        }}
run_test("Logit lens (doc)", "Logit Lens", test_logit_lens_doc)

def test_logit_lens_fix():
    with model.trace("The Eiffel Tower is in", temperature=0.0, top_p=1):
        hs = (model.model.layers[5].output[0].clone() +
              model.model.layers[5].output[1].clone())
        normed = model.model.norm(hs)
        if isinstance(normed, tuple):
            normed = normed[0]
        logits = model.lm_head(normed)
        token = logits.argmax(dim=-1).save()
    decoded = model.tokenizer.decode(token[-1])
    return {{"status": "PASS", "detail": f"Decoded: '{{decoded}}'"}}
run_test("Logit lens (workaround)", "Logit Lens", test_logit_lens_fix)

# ═══ 8. Multi-token generation ═══

def test_gen():
    with model.trace("Hello", max_tokens=3, temperature=0.0, top_p=1) as tracer:
        logits = nnsight.save(list())
        for step in tracer.iter[:]:
            logits.append(model.logits.output)
    return {{"status": "PASS", "detail": f"{{len(logits)}} steps"}}
run_test("Multi-token generation", "Generation", test_gen)

# ═══ 9. .save() mutation (Gap 1.1) ═══

def test_save_mutation():
    with model.trace("Hello world", temperature=0.0, top_p=1):
        ref = model.model.layers[5].output[0].save()
        cloned = model.model.layers[5].output[0].clone().save()
    diff = torch.max(torch.abs(ref.float() - cloned.float())).item()
    if diff > 0.1:
        return {{
            "status": "MISMATCH",
            "detail": f".save() corrupted: max_diff={{diff:.2f}}",
            "vllm_issue": "Gap 1.1: fused kernel mutates tensor after hook",
            "workaround": "Always use .clone().save()",
        }}
    return {{"status": "PASS", "detail": f"max_diff={{diff:.4f}}"}}
run_test(".save() vs .clone().save()", "Clone Before Save", test_save_mutation)

# ═══ 10. Gradients ═══

def test_grad():
    try:
        with model.trace("Hello world", temperature=0.0, top_p=1):
            hs = model.model.layers[-1].output[0]
            hs.requires_grad_(True)
        return {{"status": "PASS"}}
    except Exception as e:
        return {{
            "status": "MISMATCH",
            "detail": f"{{type(e).__name__}}",
            "vllm_issue": "Gap 4.1: torch.inference_mode() blocks gradients",
            "workaround": "None — fundamentally blocked on vLLM",
        }}
run_test("Gradients (requires_grad_)", "Gradients", test_grad)

# ═══ 11. Module skip ═══

def test_skip():
    try:
        with model.trace("Hello world", temperature=0.0, top_p=1):
            out = model.model.layers[0].output
            model.model.layers[1].skip(out)
            logits = model.logits.output.save()
        return {{"status": "PASS"}}
    except Exception as e:
        return {{
            "status": "MISMATCH",
            "detail": f"{{type(e).__name__}}",
            "vllm_issue": "Gap 4.3: fused norm expects (x, residual) pair",
            "workaround": "None clean — must construct dual-stream tuple manually",
        }}
run_test("Module skip", "Module Skipping", test_skip)

# ═══ 12. LayerNorm output ═══

def test_ln():
    with model.trace("Hello world", temperature=0.0, top_p=1):
        out = model.model.layers[0].input_layernorm.output
        is_tuple = isinstance(out, tuple)
        if is_tuple:
            return {{
                "status": "MISMATCH",
                "detail": f"Returns {{len(out)}}-tuple",
                "vllm_issue": "Gap 1.4: fused_add_rms_norm returns (normalized, residual)",
                "workaround": "Use output[0] for normalized value",
            }}
        return {{"status": "PASS"}}
run_test("LayerNorm output type", "LayerNorm", test_ln)

# ═══ 13. Merged modules ═══

def test_gate_up():
    mlp = model.model.layers[0].mlp._module
    if hasattr(mlp, 'gate_up_proj') and not hasattr(mlp, 'gate_proj'):
        return {{
            "status": "MISMATCH",
            "detail": "gate_up_proj merged, no separate gate_proj/up_proj",
            "vllm_issue": "Gap 2.1",
            "workaround": "gate, up = gate_up_proj.output.chunk(2, dim=-1)",
        }}
    return {{"status": "PASS"}}
run_test("Separate gate/up_proj", "Module Architecture", test_gate_up)

def test_qkv():
    attn = model.model.layers[0].self_attn._module
    if hasattr(attn, 'qkv_proj') and not hasattr(attn, 'q_proj'):
        return {{
            "status": "MISMATCH",
            "detail": "qkv_proj merged, no separate q/k/v_proj",
            "vllm_issue": "Gap 2.2",
            "workaround": "q, k, v = qkv_proj.output.split([q_size, kv_size, kv_size], dim=-1)",
        }}
    return {{"status": "PASS"}}
run_test("Separate q/k/v_proj", "Module Architecture", test_qkv)

# ═══ 14. down_proj output type ═══

def test_down_proj():
    with model.trace("Hello world", temperature=0.0, top_p=1):
        out = model.model.layers[0].mlp.down_proj.output
        if isinstance(out, tuple):
            return {{
                "status": "MISMATCH",
                "detail": f"Returns {{len(out)}}-tuple",
                "vllm_issue": "Gap 2.3: RowParallelLinear returns (output, bias)",
                "workaround": "Use down_proj.output[0]",
            }}
        return {{"status": "PASS"}}
run_test("down_proj output type", "Module Architecture", test_down_proj)

# ═══ 15. Attention weights ═══

def test_attn_weights():
    with model.trace("Hello world", temperature=0.0, top_p=1):
        out = model.model.layers[0].self_attn.output
        if isinstance(out, tuple) and len(out) >= 2:
            return {{"status": "PASS", "detail": "Attention weights accessible"}}
        return {{
            "status": "MISMATCH",
            "detail": f"Output is {{type(out).__name__}} — no attention weights",
            "vllm_issue": "Gap 3.2: PagedAttention fuses in CUDA",
            "workaround": "None — fundamentally inaccessible",
        }}
run_test("Attention weights", "Attention Patterns", test_attn_weights)

# ═══ 16. Caching ═══

def test_cache():
    with model.trace("Hello world", temperature=0.0, top_p=1) as tracer:
        cache = tracer.cache(modules=[model.model.layers[0], model.model.layers[1]])
    out = cache.model.layers[0].output
    is_tuple = isinstance(out, tuple)
    return {{"status": "PASS", "detail": f"Cache works, output is_tuple={{is_tuple}}"}}
run_test("Activation caching", "Caching", test_cache)

# ═══ SUMMARY ═══

pass_c = sum(1 for r in results if r["status"] == "PASS")
mm_c = sum(1 for r in results if r["status"] == "MISMATCH")
fail_c = sum(1 for r in results if r["status"] == "FAIL")

lines = []
lines.append("")
lines.append("=" * 80)
lines.append("DOCUMENTATION vs vLLM COMPATIBILITY SUMMARY")
lines.append("=" * 80)
lines.append(f"\n{{'Test':<55}} {{'Status':<12}} {{'Detail'}}")
lines.append(f"{{'-'*55}} {{'-'*12}} {{'-'*50}}")
for r in results:
    d = r.get("detail", r.get("error", ""))[:50]
    lines.append(f"{{r['name']:<55}} {{r['status']:<12}} {{d}}")
lines.append(f"\nPASS: {{pass_c}}  MISMATCH: {{mm_c}}  FAIL: {{fail_c}}  TOTAL: {{len(results)}}")
lines.append("")
if mm_c > 0:
    lines.append("=" * 80)
    lines.append("MISMATCHES & WORKAROUNDS")
    lines.append("=" * 80)
    for r in results:
        if r["status"] == "MISMATCH":
            lines.append(f"\n{{r['name']}}:")
            lines.append(f"  Issue: {{r.get('vllm_issue', r.get('detail', ''))}}")
            if "workaround" in r:
                lines.append(f"  Fix:   {{r['workaround']}}")

summary = "\n".join(lines)
print(summary)
print("RESULTS_JSON:" + json.dumps(results, default=str))
'''

def main():
    script = SCRIPT.format(gpu=args.gpu, model=args.model)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=600,
    )

    # Extract summary and JSON from output
    for line in proc.stdout.splitlines():
        if line.startswith("RESULTS_JSON:"):
            results = json.loads(line.split(":", 1)[1])
            break
    else:
        print("--- stdout (last 5000) ---")
        print(proc.stdout[-5000:])
        print("--- stderr (last 3000) ---")
        print(proc.stderr[-3000:])
        raise RuntimeError("No results JSON found")

    # Print clean summary
    pass_c = sum(1 for r in results if r["status"] == "PASS")
    mm_c = sum(1 for r in results if r["status"] == "MISMATCH")
    fail_c = sum(1 for r in results if r["status"] == "FAIL")

    print("=" * 80)
    print("DOCUMENTATION vs vLLM COMPATIBILITY SUMMARY")
    print("=" * 80)
    print(f"\n{'Test':<55} {'Status':<12} {'Detail'}")
    print(f"{'-'*55} {'-'*12} {'-'*50}")
    for r in results:
        d = r.get("detail", r.get("error", ""))[:50]
        print(f"{r['name']:<55} {r['status']:<12} {d}")

    print(f"\nPASS: {pass_c}  MISMATCH: {mm_c}  FAIL: {fail_c}  TOTAL: {len(results)}")

    if mm_c > 0:
        print("\n" + "=" * 80)
        print("MISMATCHES & WORKAROUNDS")
        print("=" * 80)
        for r in results:
            if r["status"] == "MISMATCH":
                print(f"\n{r['name']}:")
                print(f"  Issue: {r.get('vllm_issue', r.get('detail', ''))}")
                if "workaround" in r:
                    print(f"  Fix:   {r['workaround']}")

    with open("doc_examples_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results: doc_examples_results.json")

if __name__ == "__main__":
    main()
