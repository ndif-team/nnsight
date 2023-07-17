from .Model import Model

model = Model('gpt2')

with model.invoke('Hello world'):

    mmlp0 = model.h[0].mlp.output.copy()
    mmlp1 = model.h[1].mlp.output + 2

    model.h[2].mlp.output = mmlp0 + mmlp1

with model.invoke('Goodbye world'):

    model.h[1].mlp.output = mmlp1

model(device='cuda:0')

breakpoint()