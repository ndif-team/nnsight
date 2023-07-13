from .models import NDIFModel

model = NDIFModel('gpt2')


with model.invoke(['Hello world'], device='cuda:0') as invoker:

    mmlp0 = model.h[0].mlp.output.save()
    mmlp1 = model.h[1].mlp.output

    model.h[2].mlp.output = mmlp0
