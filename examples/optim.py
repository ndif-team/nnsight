from typing import Any

import torch
from engine import AbstractModel, LanguageModel, util
from engine.Module import Module
from torch.utils.data import DataLoader, Dataset

model = LanguageModel("gpt2", device_map="cuda:0")

n_tokens = 10
epochs = 20
answer = "Paris"
answer_tokens = model.tokenizer(answer)
answer_token = answer_tokens["input_ids"][0]


class SoftPrompt:
    def __init__(self, module: Module, n: int) -> None:
        self.module = module
        self.n = n

        self.embedding = torch.nn.Parameter(
            torch.zeros((self.n, self.module.embedding_dim)), requires_grad=True
        )

    def __call__(self) -> Any:
        self.module.output = self.embedding[:]

    def parameters(self):
        return [self.embedding]


sp = SoftPrompt(model.transformer.wte, n_tokens)

optimizer = torch.optim.AdamW(sp.parameters())
dataset = [[" ".join(["_"] * n_tokens), answer_token]] * 100
dataloader = DataLoader(dataset, batch_size=10)

for epoch in range(epochs):
    print(epoch)

    for i, (inputs, targets) in enumerate(dataloader):
        print(f"  {i}")

        optimizer.zero_grad()

        with model.forward(inputs, inference=False) as runner:
            sp()

            logits = model.lm_head.output.save()

        output = runner.output

        loss = util.cross_entropy_loss(output[0][:, -1], targets)
        loss.backward()

        optimizer.step()


with model.generate() as generator:
    with generator.invoke(dataset[0][0]) as invoker:
        pass

print(model.tokenizer.decode(generator.output[0]))


with model.generate() as generator:
    with generator.invoke(dataset[0][0]) as invoker:
        sp()

print(model.tokenizer.decode(generator.output[0]))
