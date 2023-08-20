from engine import Model

model = Model("gpt2")

with model.invoke('Hello world') as invoker:
    
    hiddenstates = model.transformer.h[2].output[0]
    value = model.lm_head(model.transformer.ln_f(hiddenstates)).save()

output = model(device_map='cuda:0')

print(value.value())
