"""This module contains the main NNsight model classes which enable the tracing and interleaving functionality of nnsight. 

Models allow users to load and wrap torch modules. Here we load gpt2 from HuggingFace using its repo id:

.. code-block:: python

    from nnsight import LanguageModel
    model = LanguageModel('gpt2', device_map='cuda:0')

In this case, declaring a LanguageModel entails the underlying model is a ``transformers.AutoModelForCausalLM``, and unused arguments by LanguageModel are passed downstream to AutoModelForCausalLM. ``device_map='cuda:0'`` leverages the accelerate package to use the first GPU when loading the local model.

The wrapping of the underlying model encompasses both the display of its module structure when the model is printed and the ability to directly access the attributes of the underlying meta_model through the wrapped model itself.

Printing out the wrapped module returns its structure:

.. code-block:: python

    print(model)

.. code-block:: text

    GPT2LMHeadModel(
      (transformer): GPT2Model(
        (wte): Embedding(50257, 768)
        (wpe): Embedding(1024, 768)
        (drop): Dropout(p=0.1, inplace=False)
        (h): ModuleList(
          (0-11): 12 x GPT2Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): GPT2AttentionAltered(
              (c_attn): Conv1D()
              (c_proj): Conv1D()
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
              (query): WrapperModule()
              (key): WrapperModule()
              (value): WrapperModule()
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): GPT2MLP(
              (c_fc): Conv1D()
              (c_proj): Conv1D()
              (act): NewGELUActivation()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
        (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (lm_head): Linear(in_features=768, out_features=50257, bias=False)
    )

The primary method of interacting and running the model is ``.trace(...)``. This returns a context manager object which, when entered, track operations performed on the inputs and outputs of modules.



The :func:`trace <nnsight.models.NNsightModel.NNsightModel.trace>` context is has the most explicit control of all levels of nnsight tracing and interleaving, creating a parent context where sub, input specific, contexts are spawned from.

.. code-block:: python

    with model.trace("The Eiffel Tower is in the city of") as runner:
        logits = model.lm_head.output.save()

    print(logits.value)

See :mod:`nnsight.contexts` for more.

"""

from .NNsightModel import NNsight
