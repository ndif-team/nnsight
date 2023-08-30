from engine import Model

model = Model("gpt2")

with model.generate(device_map='cuda:0') as generator:
    with generator.invoke('Hello world') as invoker:
        
        hiddenstates = model.transformer.h[2].output[0]
        value = model.lm_head(model.transformer.ln_f(hiddenstates)).save()

print(value.value)
