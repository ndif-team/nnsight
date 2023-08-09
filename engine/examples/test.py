# The library is called engine
from engine import Model
import torch
# Get model wrapper for any model you can get with AutoConfig.from_pretrained(model_name)
model = Model('gpt2')

# Prints normal module tree to show access tree for modules
print(model)

# Invoke using a prompt
with model.invoke('Hello world') as invoker:

    # See the input prompt seperated into token strings
    tokenized = invoker.tokens
    hello, _world = tokenized

    # Use normal module access and .output to get output activations.
    # Then save the activations at this point in the execution tree
    # not only to retrieve later on (by calling mlp0.value after the model has been ran),
    # but if this value changes throughout interventions and you want the value before those alteration.
    #
    # Does not work with .input (yet)
    mlp0  = model.transformer.h[0].mlp.output.copy()

    # Copy the activations, sliced by the first token
    mlp0_t1 = model.transformer.h[0].mlp.output[:,0].copy()
    # Or, use .token[token] or .t[token] to index it for you!
    mlp0_t1_t = model.transformer.h[0].mlp.output.t[hello].copy()

    mmlp0 = model.transformer.h[0].mlp.output
    mmlp1 = model.transformer.h[1].mlp.output 
    # Addition works like you normally would either with tensors or primatives ( will add other operations later)
    noise = (0.001**0.5)*torch.randn(mmlp1.t[1].shape)
    mmlp1 = mmlp1.t[1] + noise

    mmlp2_before = model.transformer.h[2].mlp.output.copy()

    # Easily set the output of a module to whatever you want
    model.transformer.h[2].mlp.output = mmlp0 + mmlp1

    # See the before and after the intervation
    mmlp2_after = model.transformer.h[2].mlp.output.copy()

with model.invoke('Goodbye world') as invoker:

    # Operations work cross-prompt!
    model.transformer.h[1].mlp.output = mmlp0

output = model(device_map='server', max_new_tokens=3)

breakpoint()
