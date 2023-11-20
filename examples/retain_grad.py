from nnsight import LanguageModel

model = LanguageModel("gpt2", device_map='cuda:0')

with model.forward(inference=False) as runner: # <-- inference mode is turned off
    with runner.invoke("testing") as invoker:
        
        logits = model.lm_head.output

        mlp = model.transformer.h[0].mlp.output.save() # <-- get mlp activations
        mlp.retain_grad() # <-- tell activations to retain it's grad

        loss = logits.sum()
        loss.backward() # <-- do backward

print(mlp.value.grad) # <-- grad is populated

with model.forward(inference=False) as runner: # <-- inference mode is turned off
    with runner.invoke("testing") as invoker:
        
        logits = model.lm_head.output

        mlp_grad = model.transformer.h[0].mlp.backward_output.save() # <-- Directly get grad

        loss = logits.sum()
        loss.backward() # <-- do backward

print(mlp_grad.value) # <-- value is populated



        