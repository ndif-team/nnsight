from engine import Model

model = Model('gpt2')

with model.invoke('Hello world') as invoker:
    
    zzz = model.transformer.h[2].output[0]
    value = model.lm_head(model.transformer.ln_f(zzz)).copy()

output = model(device='cuda:0')

breakpoint()
