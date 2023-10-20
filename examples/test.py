# The library is called nnsight
from nnsight import LanguageModel
import torch
# Get model wrapper for any model you can get with AutoConfig.from_pretrained(model_name)
model = LanguageModel('gpt2',device_map='cuda:0')

# Prints normal module tree to show access tree for modules
print(model)

# Invoke using a prompt
with model.generate(max_new_tokens=3) as generator:
    with generator.invoke('Hello world ') as invoker:

        # See the input prompt seperated into token strings
        tokenized = invoker.tokens

        # Use normal module access and .output to get output activations.
        # Then save the activations at this point in the execution tree
        # not only to retrieve later on (by calling mlp0.value after the model has been ran),
        # but if this value changes throughout interventions and you want the value before those alterations.

        mlp0  = model.transformer.h[0].mlp.output.save()

        # Use .token[idx] or .t[idx] to index by token
        mlp0_t1_t = model.transformer.h[0].mlp.output.t[0].save()

        mmlp0 = model.transformer.h[0].mlp.output
        mmlp1 = model.transformer.h[1].mlp.output 
        # Addition works like you normally would either with tensors or primatives ( will add other operations later)
        noise = (0.001**0.5)*torch.randn(mmlp1.t[1].shape)
        mmlp1 = mmlp1.t[1] + noise

        mmlp2_before = model.transformer.h[2].mlp.output.save()

        # Easily set the output of a module to whatever you want
        model.transformer.h[2].mlp.output = mmlp0 + mmlp1

        # See the before and after the intervation
        mmlp2_after = model.transformer.h[2].mlp.output.save()

    with generator.invoke('Goodbye world') as invoker:

        # Operations work cross-prompt!
        model.transformer.h[1].mlp.output = mmlp0

