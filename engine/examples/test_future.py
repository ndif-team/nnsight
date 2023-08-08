from engine import Model

model = Model('gpt2')

with model.invoke('Madison Square Garden is located in the city of') as invoker:

    print(invoker.tokens)
    
    hs = model.transformer.h[-2].output[0].copy()

    zzz = model.transformer.h[-1]()

output = model(device_map='cuda:0')

breakpoint()
