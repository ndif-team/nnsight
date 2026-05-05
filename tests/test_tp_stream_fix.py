"""
Tests for the CUDA stream propagation fix (issue #631).

Verifies that propagating the caller's CUDA stream to mediator worker
threads produces deterministic and correct intervention results under
tensor parallelism.

For thorough multi-run determinism testing, see tests/repro_631.py.

Usage:
    pytest tests/test_tp_stream_fix.py --tp 2 -v
"""

import pytest
import torch

try:
    from nnsight.modeling.vllm import VLLM
except Exception as e:
    pytest.skip(f"Skipping VLLM tests: \n{e}", allow_module_level=True)


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The Eiffel Tower is in the city of"
TARGET_TOKEN = " Paris"


@pytest.fixture(scope="module")
def tp(request):
    tp = request.config.getoption("--tp")
    if tp > torch.cuda.device_count() or tp < 1:
        pytest.exit("--tp can't be higher than the number of available GPUs.")
    return tp


@pytest.fixture(scope="module")
def model(tp: int):
    return VLLM(
        MODEL,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.1,
        dispatch=True,
        dtype=torch.float16,
    )


@pytest.fixture(scope="module")
def token_id(model):
    ids = model.tokenizer.encode(TARGET_TOKEN, add_special_tokens=False)
    assert len(ids) == 1
    return ids[0]


class TestTPStreamFix:
    """Tests for issue #631: TP>1 intervention determinism via stream propagation."""

    @torch.no_grad()
    def test_worker_stream_not_null(self, tp, model):
        """Under TP>1, worker thread should use vLLM's non-default stream."""
        if tp < 2:
            pytest.skip("Stream mismatch only occurs with TP>1")

        import nnsight

        with model.trace(temperature=0.0, max_tokens=1) as tracer:
            with tracer.invoke(PROMPT):
                worker_ptr = nnsight.save(torch.cuda.current_stream().cuda_stream)

        assert worker_ptr != 0, (
            f"Worker thread is on the NULL stream ({worker_ptr:#x}). "
            f"Stream propagation fix is not working."
        )

    @torch.no_grad()
    def test_head_ablation_determinism(self, tp, model, token_id):
        """Head ablation on o_proj.input should produce identical results across 2 runs."""
        if tp < 2:
            pytest.skip("This test targets TP>1 intervention determinism")

        head_dim = model.model.config.hidden_size // model.model.config.num_attention_heads
        num_heads = model.model.config.num_attention_heads
        layer = model.model.config.num_hidden_layers - 4
        heads = list(range(num_heads))

        runs = []
        for _ in range(2):
            with model.trace(temperature=0.0, max_tokens=1) as tracer:
                logits_list = list().save()
                for h in heads:
                    with tracer.invoke(PROMPT):
                        head_in = model.model.layers[layer].self_attn.o_proj.input.clone()
                        head_in[:, h * head_dim:(h + 1) * head_dim] = 0
                        model.model.layers[layer].self_attn.o_proj.input = head_in
                        logits_list.append(model.logits[-1, token_id].item().save())
            runs.append(logits_list[:])

        for i, h in enumerate(heads):
            assert runs[0][i] == runs[1][i], (
                f"Head {h}: run 0 = {runs[0][i]}, run 1 = {runs[1][i]}"
            )
