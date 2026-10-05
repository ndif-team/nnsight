"""PEFT adapters on a [`TransformersModel`][nnsight.modeling.transformers.TransformersModel].

An adapter is attached to the model it was given, in place (`attach`): each
targeted module becomes the adapter's layer, holding the original as
``base_layer``, and no other module moves. So a path into the base model
names the same module with or without an adapter, and `detach` puts the
held modules back. `swap` moves a loaded model from one adapter to another
and rebuilds its envoy tree, which is what ``model.load_adapter`` and a
per-request adapter on the server go through.

``peft`` is an optional dependency, imported on demand (`import_peft`).
"""

from __future__ import annotations

from typing import Optional

import torch


def import_peft():
    """Import ``peft`` on demand (it's an optional dependency).

    Only needed when a ``peft=<repo_id>`` adapter is requested, so importing it
    lazily keeps nnsight usable without peft installed.
    """
    try:
        import peft
    except ImportError as error:
        raise ImportError(
            "Using `peft=<repo_id>` requires the optional `peft` package, which "
            "is not installed. Install it with `pip install peft`."
        ) from error
    return peft

def attach(model: torch.nn.Module, peft_id: str, weights: bool = True) -> None:
    """Attach a PEFT adapter to ``model`` in place, from its repo id or path.

    The adapter's layers are injected into the model it was given, where
    ``PeftModel.from_pretrained`` would wrap it: each targeted module becomes
    the adapter's layer holding the original as ``base_layer``, and every other
    module stays where it was. So a path into the base model
    (``transformer.h.0.attn``) names the same module with or without an adapter.

    ``weights=False`` reads only the adapter's config, which is all a meta model
    needs: it gains the adapter's modules and paths, and no weights are loaded.

    An adapter whose weights do not land is refused. peft places them by
    **name** and leaves the ones it cannot match at their initialisation, and
    as ``lora_B`` starts at zero that adapter is exactly the identity: a
    base-vs-adapter comparison silently becomes base-vs-base with every number
    in it plausible.
    """
    peft = import_peft()
    config = peft.PeftConfig.from_pretrained(peft_id)
    if config.is_prompt_learning or getattr(config, "is_adaption_prompt", False):
        raise ValueError(
            f"The PEFT adapter {peft_id!r} is a {config.peft_type} adapter, which "
            "works by wrapping the model's forward and cannot be attached to its "
            "modules in place. Adapters that add layers (LoRA and its relatives) "
            "are the ones `peft=` and `TransformersModel.load_adapter` take."
        )
    from peft.functional import cast_adapter_dtype

    base = {id(parameter) for parameter in model.parameters()}
    peft.inject_adapter_in_model(config, model)
    # What `PeftModel` does on construction: a half-precision adapter is held
    # in float32.
    cast_adapter_dtype(model, "default")
    if not weights:
        return
    # The adapter's own parameters: a wrapped module's keep their identity but
    # not their names (`c_attn.weight` is now `c_attn.base_layer.weight`).
    added = {name for name, parameter in model.named_parameters() if id(parameter) not in base}

    # A saved adapter names its weights from peft's wrapper.
    state = {
        key.removeprefix("base_model.model."): value
        for key, value in peft.utils.load_peft_weights(peft_id, device="cpu").items()
    }
    result = peft.set_peft_model_state_dict(model, state)
    missing = [key for key in result.missing_keys if key in added]
    if missing or result.unexpected_keys:
        detach(model)
        unplaced = (list(result.unexpected_keys) or missing)[:3]
        raise ValueError(
            f"The PEFT adapter {peft_id!r} did not attach: peft could not match its "
            "weights to this model's modules by name, so it would be a no-op and "
            "the model would behave exactly like the base checkpoint. The usual "
            "cause is a `task=` that builds a different architecture than the "
            "adapter was trained against: e.g. task='text-generation' where the "
            "adapter targets a multimodal config, which needs "
            f"task='image-text-to-text'. Weights with no place: {unplaced}"
        )


def detach(model: torch.nn.Module) -> None:
    """Take a PEFT adapter attached by `attach` back out, in place.

    Every adapter layer is replaced by the module it holds, which leaves the
    model with the modules and paths it had before.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer
    from peft.utils.other import AuxiliaryTrainingWrapper

    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, BaseTunerLayer):
                setattr(parent, name, child.get_base_layer())
            elif isinstance(child, AuxiliaryTrainingWrapper):
                # `modules_to_save` and trainable tokens wrap a whole module.
                setattr(parent, name, child.original_module)
    if hasattr(model, "peft_config"):
        del model.peft_config

def swap(model, requested: Optional[str]) -> None:
    """Swap ``model``'s loaded PEFT adapter to ``requested``, and rebuild its envoy tree.

    The module changes only when the requested adapter differs from the
    current one, so a repeat request pays nothing:

        current  requested  action
        -------  ---------  ------
        None     None       no-op
        None     X          attach X
        X        X          no-op
        X        Y          detach X, attach Y
        X        None       detach X
    """
    if requested == model.peft:
        return

    previous = set(model._module.modules())
    if model.peft is not None:
        detach(model._module)
        # The module is the base checkpoint from here; keep `model.peft` honest
        # so a refused load below leaves the model self-consistent.
        model.peft = None
    try:
        if requested:
            # Refused before the tree is rebuilt, so a no-op adapter is never
            # what it describes. A swap is where that matters most: sweeping
            # several adapters over one loaded base is exactly the workload
            # where every organism silently collapsing to the base
            # checkpoint looks like a real result.
            attach(model._module, requested)
            model.peft = requested
    finally:
        model._rebind(model._module, previous)
