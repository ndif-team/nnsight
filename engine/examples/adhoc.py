from engine import Model

model = Model("EleutherAI/gpt-j-6B")

with model.invoke('Hello world') as invoker:
    
    zzz = model.transformer.h[2].output[0]
    value = model.lm_head(model.transformer.ln_f(zzz)).copy()

output = model(device_map='cuda:0')

breakpoint()
