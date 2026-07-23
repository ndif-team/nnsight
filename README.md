<p align="center">
  <img src="./nnsight_logo.svg" alt="nnsight" width="300">
</p>

<h3 align="center">
Interpret and manipulate the internals of deep learning models
</h3>

<p align="center">
<a href="https://www.nnsight.net"><b>Documentation</b></a> | <a href="https://github.com/ndif-team/nnsight"><b>GitHub</b></a> | <a href="https://discord.gg/6uFJmCSwW7"><b>Discord</b></a> | <a href="https://discuss.ndif.us/"><b>Forum</b></a> | <a href="https://x.com/ndif_team"><b>Twitter</b></a> | <a href="https://arxiv.org/abs/2407.14561"><b>Paper</b></a>
</p>

<p align="center">
<a href="https://colab.research.google.com/github/ndif-team/nnsight/blob/main/NNsight_Walkthrough.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></img></a>
<a href="https://deepwiki.com/ndif-team/nnsight"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></img></a>
</p>

---

## About

**nnsight** lets you get inside a model's forward pass. Open a `with model.trace(...)`
block and write ordinary Python against any internal value — a layer's output, an
attention pattern, a gradient — as if you already had it: read it, edit it, save it.
You don't register hooks or refactor the model; you write the intervention in the
order it happens, and nnsight runs it interleaved with the real forward pass.

The same trace runs on a model on your laptop or, with `remote=True`, on a model far
too large for it via the [NDIF](https://ndif.us/) infrastructure. nnsight works with
any PyTorch model and ships wrappers for HuggingFace, diffusers, and vLLM.

> 📖 For how it works under the hood — tracing, interleaving, the envoy tree — read
> **[NNsight.md](./NNsight.md)**. Task recipes live under [`docs/`](docs/).

## Installation

```bash
pip install nnsight
```

## Quick start

```python
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("The Eiffel Tower is in the city of"):
    # read a hidden state (a [batch, seq, hidden] tensor)
    hidden = model.transformer.h[6].output.save()

    # edit a layer's output in place — the model computes on the edited value
    model.transformer.h[0].output[:] = 0

    # keep the final logits
    logits = model.output.logits.save()

print(hidden.shape)            # torch.Size([1, 10, 768])
print(logits.argmax(-1))       # next-token predictions
```

Inside the block you're not running the model — you're describing what to do when it
runs. Reading `.output` gives you the real tensor once the model reaches that module;
assigning to it splices your value in. Mark anything you want after the block with
`.save()` (or `nnsight.save(x)`).

> A GPT-2 block's `.output` is a plain tensor; some modules (like attention) return a
> tuple, so you'd index `.output[0]`. `print(model)` or `print(module.source)` shows
> the shape.

## What you can do

**Generate** — `generate` returns token ids on `tracer.result`:

```python
with model.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
    ids = tracer.result.save()
print(model.tokenizer.decode(ids[0]))     # "The Eiffel Tower is in the middle of"
```

**Reach into each generation step** — save a container, append raw values (use a
bounded range so code after the loop still runs):

```python
import nnsight

with model.generate("Hello", max_new_tokens=5) as tracer:
    tokens = nnsight.save([])
    for step in tracer.iter[:5]:
        tokens.append(model.output.logits[0, -1].argmax(-1))
```

**Batch several prompts in one pass** — each `invoke` block sees only its own rows:

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        eiffel = model.transformer.h[-1].output[:, -1].save()
    with tracer.invoke("The Great Wall is in"):
        wall = model.transformer.h[-1].output[:, -1].save()
```

**Take gradients** with respect to an internal value:

```python
with model.trace("The Eiffel Tower is in the city of"):
    hidden = model.transformer.h[-1].output
    with model.output.logits.sum().backward():
        grad = hidden.grad.save()
```

**Apply modules out of order** (a logit lens — decode a middle layer through the head):

```python
with model.trace("The Eiffel Tower is in the city of"):
    hidden = model.transformer.h[-1].output
    token = model.lm_head(model.transformer.ln_f(hidden))[0, -1].argmax(-1).save()
print(model.tokenizer.decode(token))      # " Paris"
```

**Run it remotely** on NDIF — the same trace, on a model you can't host:

```python
from nnsight import CONFIG
CONFIG.set_default_api_key("YOUR_NDIF_KEY")

model = TransformersModel("meta-llama/Llama-3.1-8B")
with model.trace("The Eiffel Tower is in", remote=True):
    hidden = model.model.layers[-1].output.save()
```

There's more — **source tracing** into a module's forward (`.source`), persistent
**`edit()`**, **`skip()`**, **`scan()`** for shapes, **`cache()`**, **`session()`**,
and the **vLLM** and **diffusion** runtimes. See [`docs/`](docs/) and
[NNsight.md](./NNsight.md).

## Your own model

Any `torch.nn.Module` works — wrap it in `NNsight` and the whole tree becomes
traceable:

```python
import torch
from nnsight import NNsight

net = torch.nn.Sequential(torch.nn.Linear(5, 10), torch.nn.Linear(10, 2))
model = NNsight(net)

with model.trace(torch.rand(1, 5)):
    model[0].output[:] = 0                 # zero the first layer's output
    out = model.output.save()

print(out)                                 # [1, 2], computed with layer 0 zeroed
```

## Using nnsight from an LLM agent

Give an agent up-to-date nnsight knowledge one of these ways:

- **Skills** — in Claude Code: `/plugin marketplace add https://github.com/ndif-team/skills.git`
  then `/plugin install nnsight@skills`. In OpenAI Codex:
  `skill-installer install https://github.com/ndif-team/skills.git`.
- **Context7 MCP** — add `use context7` to prompts, or point your MCP client at
  `https://mcp.context7.com/mcp` (see [Context7](https://github.com/upstash/context7)).
- **Docs in context** — hand the agent [CLAUDE.md](./CLAUDE.md) (routes to the
  task docs) and [NNsight.md](./NNsight.md) (the internals).

## Learn more

- **[nnsight.net](https://www.nnsight.net)** — tutorials, guides, API reference
- **[NNsight.md](./NNsight.md)** — the design-and-implementation manual
- **[CLAUDE.md](./CLAUDE.md)** + **[docs/](docs/)** — the task reference
- **[nnsight.net/status](https://nnsight.net/status/)** — models available on NDIF

## Citation

If you use `nnsight` in your research, please cite:

```bibtex
@article{fiottokaufman2024nnsightndifdemocratizingaccess,
      title={NNsight and NDIF: Democratizing Access to Foundation Model Internals},
      author={Jaden Fiotto-Kaufman and Alexander R Loftus and Eric Todd and Jannik Brinkmann and Caden Juang and Koyena Pal and Can Rager and Aaron Mueller and Samuel Marks and Arnab Sen Sharma and Francesca Lucchetti and Michael Ripa and Adam Belfki and Nikhil Prakash and Sumeet Multani and Carla Brodley and Arjun Guha and Jonathan Bell and Byron Wallace and David Bau},
      year={2024},
      eprint={2407.14561},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.14561},
}
```
