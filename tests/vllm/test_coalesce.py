"""Module calls and replacements of several requests of one trace.

With coalescing on (the default) the requests' token-wise calls of one module
run as one call and their replacements at one layer are written in one pass.
Which calls are token-wise is decided by the module's class
(``modeling/vllm/coalesce.py``). The engine tests compare a request's result in
a trace of several requests with the same request traced alone, where nothing
can be shared, on one rank and across ranks.
"""

import pytest
import torch

PROMPTS = [
    "The Eiffel Tower is located in the city of",
    "Madison Square Garden is located in the city of",
    "Water boils at a temperature of one hundred degrees",
    "The capital of Japan is",
]
SAMPLING = dict(temperature=0.0, top_p=1)


def _lens(model):
    """The last token's logits after every layer, for the request being traced."""
    rows = []
    for layer in model.model.layers:
        out = layer.output
        normed = model.model.norm((out[0] + out[1])[-1:, :])
        rows.append(model.logits_processor(model.lm_head, normed))
    return torch.stack(rows)


def _steer(model):
    """Add a constant to two layers' hidden states; the request's final logits."""
    for index in (1, 3):
        out = model.model.layers[index].output
        model.model.layers[index].output = (out[0] + 0.5, *out[1:])
    return model.logits.clone()


BODIES = pytest.mark.parametrize("body", [_lens, _steer], ids=["module-calls", "replacements"])


@BODIES
def test_requests_of_one_trace_match_each_traced_alone(vllm_qwen_ref, body):
    _match(vllm_qwen_ref, body)


@BODIES
def test_sharded_requests_of_one_trace_match_each_traced_alone(vllm_qwen_tp, body):
    _match(vllm_qwen_tp, body)


def test_the_predicate_decides_by_class_and_call_form():
    from types import SimpleNamespace

    from vllm.model_executor.layers.layernorm import RMSNorm

    from nnsight.intervention.interleaver import Call
    from nnsight.modeling.vllm.coalesce import tokenwise

    class Projection(torch.nn.Linear):
        pass

    class Pool(torch.nn.Module):
        def forward(self, hidden):
            return hidden.mean(dim=0, keepdim=True)

    eligible = tokenwise()
    rows = torch.zeros(3, 8)

    def call(module, *args, **kwargs):
        return Call(SimpleNamespace(_module=module), args, kwargs)

    for module in (torch.nn.Linear(8, 4), Projection(8, 4), torch.nn.LayerNorm(8), torch.nn.Embedding(5, 8)):
        assert eligible(call(module, rows))
    # Only its class is read; building one needs an engine's config.
    norm = RMSNorm.__new__(RMSNorm)
    assert eligible(call(norm, rows))
    # With a residual it adds and normalizes in place.
    assert not eligible(call(norm, rows, rows))
    assert not eligible(call(norm, rows, residual=rows))
    # Modules that mix rows, and containers, whatever they hold.
    for module in (Pool(), torch.nn.MultiheadAttention(8, 2), torch.nn.Sequential(torch.nn.Linear(8, 8))):
        assert not eligible(call(module, rows))


@torch.no_grad()
def test_a_norm_is_shared_alone_and_not_with_its_residual(vllm_qwen_ref):
    qwen = vllm_qwen_ref
    # Layer 1's own forward calls this norm with a residual, on a whole prompt's
    # rows when the request ends after one token; so every call with the hidden
    # state alone, and every call on one row, is a block's.
    once = dict(SAMPLING, max_tokens=1)
    with qwen.trace(PROMPTS[0], **once):
        module = qwen.model.layers[1].input_layernorm._module
        module.seen = []
        module.watch = module.register_forward_pre_hook(
            lambda module, args: module.seen.append((len(args), args[0].shape[0]))
        )

    with qwen.trace(**once) as tracer:
        wrote = [None] * len(PROMPTS)
        wrote = wrote.save()
        for index, prompt in enumerate(PROMPTS):
            with tracer.invoke(prompt):
                out = qwen.model.layers[2].output
                hidden = (out[0] + out[1])[-1:, :].clone()
                norm = qwen.model.layers[1].input_layernorm
                pure = norm(hidden)
                given, residual = hidden.clone(), torch.ones_like(hidden)
                fused, summed = norm(given, residual)
                # The fused form leaves the sum in the caller's own residual.
                wrote[index] = bool(
                    torch.equal(residual, summed)
                    and not torch.equal(residual, torch.ones_like(hidden))
                    and torch.allclose(fused.float(), norm(summed.clone()).float(), atol=1e-2)
                    and pure.shape == hidden.shape
                )

    with qwen.trace(PROMPTS[0], **once):
        module = qwen.model.layers[1].input_layernorm._module
        seen = list(module.seen).save()
        module.watch.remove()
        del module.seen, module.watch

    assert all(wrote)
    alone = [rows for count, rows in seen if count == 1]
    # Two one-argument calls per request, one row each: fewer calls than that,
    # every row accounted for.
    assert sum(alone) == 2 * len(PROMPTS) and len(alone) < 2 * len(PROMPTS)
    # The fused form: one call per request, on that request's one row.
    assert [rows for count, rows in seen if count == 2].count(1) == len(PROMPTS)


def test_an_error_in_one_requests_call_leaves_the_others(vllm_qwen_ref):
    _one_error(vllm_qwen_ref)


def test_an_error_in_one_sharded_requests_call_leaves_the_others(vllm_qwen_tp):
    _one_error(vllm_qwen_tp)


@torch.no_grad()
def _match(qwen, body):
    with qwen.trace(**SAMPLING) as tracer:
        together = [None] * len(PROMPTS)
        together = together.save()
        for index, prompt in enumerate(PROMPTS):
            with tracer.invoke(prompt):
                together[index] = body(qwen)

    for index, prompt in enumerate(PROMPTS):
        with qwen.trace(prompt, **SAMPLING):
            alone = body(qwen).save()
        ours = together[index].float().cpu()
        alone = alone.float().cpu()
        assert ours.shape == alone.shape
        # bf16 arithmetic depends on the batch a request ran in, so a nearly
        # tied argmax can differ; the values agree closely.
        cosine = torch.nn.functional.cosine_similarity(
            ours.flatten(0, -2), alone.flatten(0, -2), dim=-1
        )
        assert cosine.min() > 0.999


@torch.no_grad()
def _one_error(qwen):
    with qwen.trace(**SAMPLING) as tracer:
        outcome = [None] * len(PROMPTS)
        outcome = outcome.save()
        for index, prompt in enumerate(PROMPTS):
            with tracer.invoke(prompt):
                out = qwen.model.layers[0].output
                hidden = (out[0] + out[1])[-1:, :]
                if index == 2:
                    hidden = hidden[:, :7]  # the norm refuses this width
                try:
                    outcome[index] = tuple(qwen.model.norm(hidden).shape)
                except Exception:
                    outcome[index] = "raised"
    assert outcome[2] == "raised"
    assert len({outcome[index] for index in (0, 1, 3)}) == 1 and outcome[0] != "raised"
