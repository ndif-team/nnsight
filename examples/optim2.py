from typing import Any

import torch
from nnsight import NNsightModel, LanguageModel, util
from nnsight.module import Module
from torch.utils.data import DataLoader, Dataset

model = LanguageModel("gpt2", device_map="cuda:0")

n_tokens = 10
epochs = 1
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


def decode(hs):
    return torch.log_softmax(model.lm_head(model.transformer.ln_f(hs)), dim=-1)

with model.generate(max_new_tokens=2) as generator:

    with generator.invoke("Madison Square Garden is located in New") as invoker:

        hs = model.transformer.h[1].output[0].t[-1].save()

        invoker.next()

        target = decode(model.transformer.h[-1].output[0]).save()

target = target.value
hs = hs.value

lossfn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

for epoch in range(epochs):
    print(epoch)

    for i, (inputs, targets) in enumerate(dataloader):
        print(f"  {i}")

        optimizer.zero_grad()

        with model.forward(inputs, inference=False) as runner:

            sp()

            model.transformer.h[1].output[0].t[-1] = hs

            pred = decode(model.transformer.h[-1].output[0]).save()



        loss = lossfn(pred.value, target)
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
