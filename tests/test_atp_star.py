"""Correctness tests for the public-API AtP* research pattern."""

from collections import OrderedDict
import torch
from nnsight import NNsight


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
