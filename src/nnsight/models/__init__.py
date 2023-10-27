"""Module containing Models, the main classes of this package which enables the tracing and interleaving functionality of nnsight.

Models allow users to load and wrap torch modules. 

Here we load 'gpt2' from huggingface using its repo id into a LanguageModel:

>>> from nnsight import LanguageModel

>>> model = LanguageModel('gpt2', device_map='cuda:0')

In this case, using a LanguageModel means the underlying model is a transformers.AutoModelForCausalLM, and unused arguments by LanguageModel are passed downstream to AutoModelForCausalLM.
``device_map='cuda:0`` ultimately leverages the accelerate package to use the first gpu when loading the local_model.

The wrapping of the underlying model is further encapsulated by both printing out it's module structure when printing the wrapped model, as well as accessing attributes on the meta_model by accessing them on the wrapped module.

Printing out the wrapped module demonstrates this like:

>>> print(model)

GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D()
          (c_proj): Conv1D()
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
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

The main two methods to interact and run the model are ``.generate(...)`` and ``.forward(...)``

Both return context manager objects which when entered, track operations you perform on the inputs and outputs of modules.

The generate context is meant for multi-iteration, and multi-input inference runs.
Arguments passed to generate determine the generation behavior, in this case to generate three tokens.
Within a generation context, invoker sub-contexts are entered using ``generator.invoke``. This is where input (or batch of inputs) to the model is accepted, and batched with other invocations.
It's in these contexts where operations on inputs and outputs of modules are tracked and prepared for execution.

In this example, we run two prompts on the language model in order to generate three tokens.
We also perform a ``.save()`` operation on the output of the lm_head module, the logit outputs, in order to save these activations and access them after generation is over:

>>> model.generate(max_new_tokens=3) as generator:
>>>     generator.invoke("The Eiffel Tower is in the city of") as invoker:
>>>         logits1 = model.lm_head.output.save()
>>>     generator.invoke("The Empire State Building is in the city of") as invoker:
>>>         logits2 = model.lm_head.output.save()
>>> print(logits1.value)
>>> print(logits2.value)
>>> print(runner.output)

The forward, runner context is meant for direct input to the underlying model (or module) in either inference or training mode where gradients are collected.

>>> model.forward("The Eiffel Tower is in the city of", inference=True) as runner:
>>>     logits = model.lm_head.output.save()
>>> print(logits.value)

See :mod:`nnsight.contexts`
"""
