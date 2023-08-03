from engine import Model

model = Model('gpt2')

with model.invoke('Hello world') as invoker:
    
    zzz = model.transformer.h[2].output[0].copy()

output = model(device_map='server')

breakpoint()
