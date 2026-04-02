"""Minimal test: does new_group work during load_model with PP=2?

Run: python tests/test_pp_new_group.py
Requires 2 GPUs.
"""

if __name__ == '__main__':
    import torch
    from nnsight.modeling.vllm import VLLM

    model = VLLM(
        "openai-community/gpt2",
        pipeline_parallel_size=2,
        gpu_memory_utilization=0.1,
        dispatch=True,
    )

    print("Model loaded successfully — new_group did not hang!")

    # Quick sanity: run a trace
    with model.trace("Hello world", temperature=0.0, top_p=1):
        logits = model.logits.output.save()

    print(f"Logits shape: {logits.shape}")
    print("PASS")
