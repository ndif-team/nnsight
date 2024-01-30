"""This module contains the main Model classes which enable the tracing and interleaving functionality of nnsight. 

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

The primary methods of interacting and running the model are ``.generate(...)`` and ``.forward(...)``. Both return context manager objects which, when entered, track operations performed on the inputs and outputs of modules.

The :func:`generate <nnsight.models.NNsightModel.NNsightModel.generate>` context is meant for multi-iteration runs. Arguments passed to generate determine the generation behavior — in this case to generate three tokens.
Within a generation context, invoker sub-contexts are entered using ``generator.invoke``. This is where an input (or batch of inputs) to the model is accepted, and batched with other invocations. It's in these contexts where operations on inputs and outputs of modules are tracked and prepared for execution.

In this example, we run two prompts on the language model in order to generate three tokens. We also perform a ``.save()`` operation on the output of the lm_head module (the logit outputs) in order to save these activations and access them after generation is over:

.. code-block:: python

    with model.generate(max_new_tokens=3) as generator:
        with generator.invoke("The Eiffel Tower is in the city of") as invoker:
            logits1 = model.lm_head.output.save()
        with generator.invoke("The Empire State Building is in the city of") as invoker:
            logits2 = model.lm_head.output.save()

    print(logits1.value)
    print(logits2.value)

The :func:`forward <nnsight.models.NNsightModel.NNsightModel.forward>` context is meant for direct input to the underlying model (or module).

.. code-block:: python

    with model.forward(inference=True) as runner:
        with runner.invoke("The Eiffel Tower is in the city of") as invoker:
            logits = model.lm_head.output.save()

    print(logits.value)

See :mod:`nnsight.contexts` for more.

"""
