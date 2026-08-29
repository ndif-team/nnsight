"""Correctness tests for the public-API AtP* research pattern."""

from collections import OrderedDict

import torch
import pytest
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
)

from nnsight import NNsight


class _CancellationModel(torch.nn.Module):
    """A direct path and an equal, opposite indirect path."""

    def __init__(self):
        super().__init__()
        self.component = torch.nn.Identity()
        self.indirect = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.indirect.weight.fill_(-1.0)

    def forward(self, inputs):
        component = self.component(inputs)
        return component + self.indirect(component)


def _attention_probabilities(query, key, mask=None, scale=None):
    scores = torch.einsum("...qd,...kd->...qk", query, key)
    if scale is None:
        scale = query.shape[-1] ** -0.5
    scores = scores * scale
    if mask is not None:
        scores = scores + mask
    return scores.softmax(dim=-1)


def _query_output_delta(
    clean_query, noise_key, noise_value, noise_output, mask=None, scale=None
):
    patched_probabilities = _attention_probabilities(
        clean_query, noise_key, mask, scale
    )
    patched_output = torch.einsum(
        "...qk,...kd->...qd", patched_probabilities, noise_value
    )
    return patched_output - noise_output


def _key_output_delta(
    noise_query,
    clean_key,
    noise_key,
    noise_value,
    noise_probabilities,
    noise_output,
    scale=None,
):
    """Exact local output deltas for every single-key patch in O(T²D)."""
    score_delta = torch.einsum("...qd,...kd->...qk", noise_query, clean_key - noise_key)
    if scale is None:
        scale = noise_query.shape[-1] ** -0.5
    score_delta = score_delta * scale

    log_odds = torch.log(noise_probabilities) - torch.log1p(-noise_probabilities)
    patched_probability = torch.sigmoid(log_odds + score_delta)
    denominator = torch.where(
        noise_probabilities == 1,
        torch.ones_like(noise_probabilities),
        1 - noise_probabilities,
    )
    probability_scale = (patched_probability - noise_probabilities) / denominator
    probability_scale = torch.where(
        noise_probabilities == 1,
        torch.zeros_like(probability_scale),
        probability_scale,
    )
    value_difference = noise_value.unsqueeze(-3) - noise_output.unsqueeze(-2)
    return probability_scale.unsqueeze(-1) * value_difference


def _attention_source_call(attention, family):
    operation_names = {
        "gpt2": ("attention_interface_0",),
        "pythia": ("attention_interface_0", "unknown_0"),
    }
    if family not in operation_names:
        raise ValueError(f"Unsupported AtP* model family: {family}")
    for name in operation_names[family]:
        try:
            return getattr(attention.source, name)
        except AttributeError:
            pass
    raise ValueError(
        f"Unsupported {family} attention source layout; expected one of "
        f"{operation_names[family]}"
    )


def _unpack_attention_call(call):
    args, kwargs = call.inputs
    offset = 0 if torch.is_tensor(args[0]) else 1
    if len(args) < offset + 3:
        raise ValueError("Attention call does not expose Q, K, and V inputs")
    query, key, value = args[offset : offset + 3]
    output, probabilities = call.output
    if output.shape == query.shape:
        normalized_output = output
    elif output.shape[-3:] == (
        query.shape[-2],
        query.shape[-3],
        query.shape[-1],
    ):
        normalized_output = output.transpose(-3, -2)
    else:
        raise ValueError("Unsupported per-head attention output layout")
    return query, key, value, normalized_output, probabilities


def _aggregate_graddrop(drop_estimates):
    """Equation 11: sum absolute layer-drop estimates and divide by L - 1."""
    layers = drop_estimates.shape[0]
    if layers < 2:
        raise ValueError("GradDrop requires at least two residual layers")
    return drop_estimates.abs().sum(dim=0) / (layers - 1)


def _subset_statistics(masks, effects):
    """Algorithm 1 statistics for included and excluded node subsets."""
    if masks.ndim != 2 or effects.shape != masks.shape[:1]:
        raise ValueError("Expected masks [sample, node] and effects [sample]")
    membership = torch.stack((masks, ~masks))
    counts = membership.sum(dim=1)
    if torch.any(counts < 2):
        raise ValueError("Each node needs two included and two excluded samples")

    expanded_effects = effects[None, :, None]
    means = (expanded_effects * membership).sum(dim=1) / counts
    centered = expanded_effects - means[:, None, :]
    variances = (centered.square() * membership).sum(dim=1) / (counts - 1)
    return counts, means, variances


def _linear_model() -> NNsight:
    """A deterministic model where first-order attribution is exact."""
    network = torch.nn.Sequential(
        OrderedDict(
            [
                ("components", torch.nn.Linear(3, 3, bias=False)),
                ("readout", torch.nn.Linear(3, 1, bias=False)),
            ]
        )
    )
    with torch.no_grad():
        network.components.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.5, -0.5],
                    [0.0, 2.0, 1.0],
                    [-1.0, 0.0, 0.25],
                ]
            )
        )
        network.readout.weight.copy_(torch.tensor([[2.0, -3.0, 0.5]]))
    return NNsight(network)


def _tiny_transformer(family):
    with torch.random.fork_rng():
        torch.manual_seed(704)
        if family == "gpt2":
            config = GPT2Config(
                vocab_size=32,
                n_positions=16,
                n_embd=8,
                n_layer=2,
                n_head=2,
            )
            config._attn_implementation = "eager"
            model = NNsight(GPT2LMHeadModel(config).eval())
            return model, model.transformer.h[0].attn

        if family == "pythia":
            config = GPTNeoXConfig(
                vocab_size=32,
                max_position_embeddings=16,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                rotary_pct=0.5,
            )
            config._attn_implementation = "eager"
            model = NNsight(GPTNeoXForCausalLM(config).eval())
            return model, model.gpt_neox.layers[0].attention

    raise ValueError(f"Unsupported AtP* model family: {family}")


def test_basic_atp_matches_exact_component_patching():
    """AtP and exact patching agree when the downstream model is linear."""
    model = _linear_model()
    clean = torch.tensor([[2.0, -1.0, 0.5]])
    noise = torch.tensor([[-1.0, 0.5, 2.0]])

    with model.trace(clean):
        clean_components = model.components.output.save()
        clean_metric = model.output.sum().save()

    with model.trace(noise):
        noise_ref = model.components.output
        noise_ref.requires_grad_(True)
        noise_components = noise_ref.save()
        noise_metric_ref = model.output.sum()
        noise_metric = noise_metric_ref.save()
        with noise_metric_ref.backward():
            noise_gradient = noise_ref.grad.clone().save()

    atp_effects = ((clean_components - noise_components) * noise_gradient)[0]
    exact_effects = []
    for component in range(clean_components.shape[-1]):
        with model.trace(noise):
            model.components.output[:, component] = clean_components[:, component]
            patched_metric = model.output.sum().save()
        exact_effects.append(patched_metric - noise_metric)
    exact_effects = torch.stack(exact_effects)

    torch.testing.assert_close(atp_effects, exact_effects)
    torch.testing.assert_close(atp_effects.sum(), clean_metric - noise_metric)
    assert torch.equal(
        atp_effects.abs().argsort(descending=True),
        exact_effects.abs().argsort(descending=True),
    )


def test_basic_atp_is_only_an_approximation_after_a_nonlinearity():
    """The test suite must not imply that AtP scores are causal effects."""
    network = torch.nn.Sequential(
        OrderedDict(
            [
                ("components", torch.nn.Linear(1, 1, bias=False)),
                ("saturation", torch.nn.Tanh()),
            ]
        )
    )
    with torch.no_grad():
        network.components.weight.fill_(1.0)
    model = NNsight(network)

    clean = torch.tensor([[0.0]])
    noise = torch.tensor([[4.0]])
    with model.trace(clean):
        clean_component = model.components.output.save()
    with model.trace(noise):
        noise_ref = model.components.output
        noise_ref.requires_grad_(True)
        noise_component = noise_ref.save()
        noise_metric_ref = model.output.sum()
        noise_metric = noise_metric_ref.save()
        with noise_metric_ref.backward():
            noise_gradient = noise_ref.grad.clone().save()
    with model.trace(noise):
        model.components.output = clean_component
        patched_metric = model.output.sum().save()

    atp_effect = ((clean_component - noise_component) * noise_gradient).sum()
    exact_effect = patched_metric - noise_metric
    assert atp_effect.abs() < exact_effect.abs() * 0.01


def test_qk_correction_matches_brute_force_attention_patches():
    """Vectorized Q/K corrections equal explicit one-node recomputation."""
    generator = torch.Generator().manual_seed(704)
    shape = (2, 3, 4, 5)
    noise_query = torch.randn(shape, generator=generator, dtype=torch.float64)
    clean_query = torch.randn(shape, generator=generator, dtype=torch.float64)
    noise_key = torch.randn(shape, generator=generator, dtype=torch.float64)
    clean_key = torch.randn(shape, generator=generator, dtype=torch.float64)
    noise_value = torch.randn(shape, generator=generator, dtype=torch.float64)
    mask = torch.full((4, 4), float("-inf"), dtype=torch.float64).triu(1)

    noise_probabilities = _attention_probabilities(noise_query, noise_key, mask)
    noise_output = torch.einsum("...qk,...kd->...qd", noise_probabilities, noise_value)

    query_delta = _query_output_delta(
        clean_query, noise_key, noise_value, noise_output, mask
    )
    for position in range(noise_query.shape[-2]):
        patched_query = noise_query.clone()
        patched_query[..., position, :] = clean_query[..., position, :]
        patched_probabilities = _attention_probabilities(patched_query, noise_key, mask)
        patched_output = torch.einsum(
            "...qk,...kd->...qd", patched_probabilities, noise_value
        )
        torch.testing.assert_close(
            query_delta[..., position, :],
            (patched_output - noise_output)[..., position, :],
        )

    key_delta = _key_output_delta(
        noise_query,
        clean_key,
        noise_key,
        noise_value,
        noise_probabilities,
        noise_output,
    )
    for position in range(noise_key.shape[-2]):
        patched_key = noise_key.clone()
        patched_key[..., position, :] = clean_key[..., position, :]
        patched_probabilities = _attention_probabilities(noise_query, patched_key, mask)
        patched_output = torch.einsum(
            "...qk,...kd->...qd", patched_probabilities, noise_value
        )
        torch.testing.assert_close(
            key_delta[..., position, :], patched_output - noise_output
        )


def test_key_correction_preserves_sub_epsilon_probabilities():
    """Small, nonzero probabilities must not be clamped before correction."""
    noise_query = torch.ones((1, 1, 1, 1), dtype=torch.float64)
    noise_key = torch.tensor([[[[0.0], [-40.0]]]], dtype=torch.float64)
    clean_key = noise_key.clone()
    clean_key[..., 1, :] = 0
    noise_value = torch.tensor([[[[2.0], [-3.0]]]], dtype=torch.float64)
    noise_probabilities = _attention_probabilities(noise_query, noise_key)
    noise_output = torch.einsum("...qk,...kd->...qd", noise_probabilities, noise_value)

    key_delta = _key_output_delta(
        noise_query,
        clean_key,
        noise_key,
        noise_value,
        noise_probabilities,
        noise_output,
    )
    patched_probabilities = _attention_probabilities(noise_query, clean_key)
    patched_output = torch.einsum(
        "...qk,...kd->...qd", patched_probabilities, noise_value
    )

    assert noise_probabilities[..., 1].item() < torch.finfo(torch.float64).eps
    torch.testing.assert_close(key_delta[..., 1, :], patched_output - noise_output)


def test_graddrop_exposes_a_cancellation_hidden_component():
    """Dropping an indirect residual gradient reveals the direct path."""
    model = NNsight(_CancellationModel())
    inputs = torch.tensor([[3.0]])

    with model.trace(inputs):
        component = model.component.output
        component.requires_grad_(True)
        indirect = model.indirect.output
        indirect.requires_grad_(True)
        metric = model.output.sum()

        with metric.backward(retain_graph=True):
            standard_gradient = component.grad.clone().save()

        with metric.backward():
            indirect.grad = torch.zeros_like(indirect.grad)
            dropped_gradient = component.grad.clone().save()

    torch.testing.assert_close(standard_gradient, torch.zeros_like(standard_gradient))
    torch.testing.assert_close(dropped_gradient, torch.ones_like(dropped_gradient))


@pytest.mark.parametrize("family", ("gpt2", "pythia"))
def test_attention_source_adapter_matches_model_attention(family):
    """Each MVP adapter reproduces the model's eager attention result."""
    input_ids = torch.tensor([[1, 2, 3, 4]])
    model, attention = _tiny_transformer(family)
    with model.trace(input_ids):
        call = _attention_source_call(attention, family)
        query, key, value, attention_output, probabilities = _unpack_attention_call(
            call
        )
        query.requires_grad_(True)
        saved_query = query.save()
        saved_key = key.save()
        saved_value = value.save()
        saved_output = attention_output.save()
        saved_probabilities = probabilities.save()
        metric = model.output.logits.float().square().mean()
        with metric.backward():
            query_gradient = query.grad.clone().save()

    causal_mask = torch.full((4, 4), float("-inf")).triu(1)
    recomputed_probabilities = _attention_probabilities(
        saved_query, saved_key, causal_mask
    )
    recomputed_output = torch.einsum(
        "...qk,...kd->...qd", recomputed_probabilities, saved_value
    )
    expected = (1, 2, 4, 4)
    assert saved_query.shape == expected
    assert saved_key.shape == expected
    assert saved_value.shape == expected
    assert saved_probabilities.shape == (1, 2, 4, 4)
    torch.testing.assert_close(saved_probabilities, recomputed_probabilities)
    torch.testing.assert_close(saved_output, recomputed_output)
    assert torch.isfinite(query_gradient).all()
    assert torch.count_nonzero(query_gradient)


def test_graddrop_aggregation_matches_equation_11():
    drop_estimates = torch.tensor(
        [
            [1.0, -2.0, 0.0],
            [-3.0, 4.0, 2.0],
            [2.0, -1.0, -4.0],
        ]
    )
    expected = torch.tensor([3.0, 3.5, 3.0])
    torch.testing.assert_close(_aggregate_graddrop(drop_estimates), expected)


def test_subset_statistics_recover_additive_node_effects():
    """Algorithm 1 is exact on a balanced, additive subset experiment."""
    nodes = 4
    integers = torch.arange(2**nodes)
    bit_positions = torch.arange(nodes)
    masks = integers[:, None].bitwise_and(1 << bit_positions).bool()
    contributions = torch.tensor([3.0, -2.0, 0.5, 4.0])
    effects = masks.float() @ contributions

    counts, means, variances = _subset_statistics(masks, effects)
    estimates = means[0] - means[1]

    assert torch.equal(counts, torch.full_like(counts, 2 ** (nodes - 1)))
    torch.testing.assert_close(estimates, contributions)
    assert torch.isfinite(variances).all()
    assert torch.all(variances >= 0)


def test_atp_star_helpers_reject_invalid_inputs():
    with pytest.raises(ValueError, match="at least two residual layers"):
        _aggregate_graddrop(torch.ones((1, 3)))

    with pytest.raises(ValueError, match="Expected masks"):
        _subset_statistics(torch.ones((2, 2), dtype=torch.bool), torch.ones((2, 1)))

    masks = torch.tensor([[True, False], [True, False]])
    with pytest.raises(ValueError, match="two included and two excluded"):
        _subset_statistics(masks, torch.ones(2))

    unsupported_attention = type("UnsupportedAttention", (), {"source": object()})()
    with pytest.raises(ValueError, match="Unsupported pythia attention source layout"):
        _attention_source_call(unsupported_attention, "pythia")
