from engine.Model import Model

model = Model('gpt2')

# Prints normal module tree
print(model)

with model.invoke('Hello world'):

    # Grab and return activations of model as well as slice with [:,0]
    sl = model.transformer.h[0].mlp.output[:,0].copy()
    # Grab another copy of activations
    mmlp0 = model.transformer.h[0].mlp.output.copy()
    mmlp1 = model.transformer.h[1].mlp.output.copy() + 2

    model.transformer.h[2].mlp.output = mmlp0 + mmlp1

with model.invoke('Goodbye world'):

    model.transformer.h[1].mlp.output = mmlp0

output = model(device='cuda:0', max_new_tokens=2)

breakpoint()