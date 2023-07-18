from .Model import Model

model = Model('gpt2')

with model.invoke('Hello world'):

    mmlp0 = model.transformer.h[0].mlp.output.copy()
    mmlp1 = model.transformer.h[1].mlp.output + 2

    model.transformer.h[2].mlp.output = mmlp0 + mmlp1

with model.invoke('Goodbye world'):

    model.transformer.h[1].mlp.output = mmlp1

output = model(device='cuda:0', max_new_tokens=1)

breakpoint()