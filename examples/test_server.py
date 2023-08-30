from engine import Model

model = Model("gpt2")

print(model)

with model.generate(device_map="server") as generator:
    with generator.invoke("Hello world") as invoker:
        hiddenstates = model.transformer.h[2].output.save()


print(hiddenstates.value)
