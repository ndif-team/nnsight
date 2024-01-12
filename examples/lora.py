from typing import Any

import torch
from nnsight import NNsightModel, LanguageModel, util
from nnsight.module import Module
from nnsight.toolbox.optim.lora import LORA
from torch.utils.data import DataLoader, Dataset

model = LanguageModel("gpt2", device_map="cuda:0")

n_tokens = 10
epochs = 1
answer = "Paris"
answer_tokens = model.tokenizer(answer)
answer_token = answer_tokens["input_ids"][0]

lora = LORA(model.transformer.h[0].mlp, 10)

optimizer = torch.optim.AdamW(lora.parameters(), lr=.1)
dataset = [[" ".join(["_"] * n_tokens), answer_token]] * 100
dataloader = DataLoader(dataset, batch_size=10)


lossfn = util.cross_entropy_loss

for epoch in range(epochs):
    print(epoch)

    for i, (inputs, targets) in enumerate(dataloader):
        print(f"  {i}")

        optimizer.zero_grad()

        with model.forward(inputs, inference=False) as runner:


            lora()

            logits = model.lm_head.output.save()

        print(lora.WA)
        loss = lossfn(logits.value[:, -1], targets)
        print(loss)

        loss.backward()

        optimizer.step()


with model.generate() as generator:
    with generator.invoke(dataset[0][0]) as invoker:
        pass

print(model.tokenizer.decode(generator.output[0]))


with model.generate() as generator:
    with generator.invoke(dataset[0][0]) as invoker:
        lora()

print(model.tokenizer.decode(generator.output[0]))
