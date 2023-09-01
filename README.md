# NDIF Repository
## engine API

The `engine/` directory contains the engine package for interpreting and manipulating the internals of large language models.

- `engine/model_checkpoints` is set to be the default huggingface hub cache directory. Contains by default models found on the NDIF server with their respective configurations.

#### Installation

Install this package through pip by running:

`pip install git+https://github.com/JadenFiotto-Kaufman/ndif`

#### Examples

Here is a simple example where we run the engine API locally on gpt2 and save the hidden states of the last layer:

```python
from engine import Model

model = Model('gpt2')

with model.generate(device_map='cuda', max_new_tokens=1) as generator:
    with generator.invoke('The Eiffel Tower is in the city of') as invoker:

        hidden_states = model.transformer.h[-1].output[0].save()

output = generator.output
hidden_states = hidden_states.value
```

Lets go over this piece by piece.

We import the `Model` object from the `engine` module and create a gpt2 model using the huggingface repo ID for gpt2, `'gpt2'`

```python
from engine import Model

model = Model('gpt2')
```

Then, we create a generation context block by calling `.generate(...)` on the model object. This denotes we wish to actually generate tokens given some prompts. `device_map='cuda` specifies running the model on the `cuda` device. 

Other keyword arguments are passed downstream to [AutoModelForCausalLM.generate(...)](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate). Refer to the linked docs for reference.


```python
with model.generate(device_map='cuda', max_new_tokens=3) as generator:
```

Now calling `.generate(...)` does not actually initialize or run the model. Only after the `with generator` block is exited, is the acually model loaded and ran. All operations in the block are "proxies" which essentially creates a graph of operations we wish to carry out later.


Within the generation context, we create invocation contexts to specify the actual prompts we want to run:


```python
with generator.invoke('The Eiffel Tower is in the city of') as invoker:
```

Within this context, all operations/interventions will be applied to the processing of this prompt.

---

Most* operations

---
###### Multiple Token Generation

When generating more than one token, use `invoker.next()` to denote following interventions should be applied to the subsequent generations.

Here we again generate using gpt2, but generate three tokens and save the hidden states of the last layer for each one:

```python
from engine import Model

model = Model('gpt2')

with model.generate(device_map='cuda', max_new_tokens=3) as generator:
    with generator.invoke('The Eiffel Tower is in the city of') as invoker:

        hidden_states1 = model.transformer.h[-1].output[0].save()

        invoker.next()
        
        hidden_states2 = model.transformer.h[-1].output[0].save()

        invoker.next()
        
        hidden_states3 = model.transformer.h[-1].output[0].save()


output = generator.output
hidden_states1 = hidden_states1.value
hidden_states2 = hidden_states2.value
hidden_states3 = hidden_states3.value
```
---

###### Token Based Indexing


When indexing hidden states for specific tokens, use `.token(<idx>)` or `.t(<idx>)`.
This is because if there are multiple invovations, padding is performed on the left side so these helper functions index from the back.

Here we just get the hidden states of the first token:

```python
from engine import Model

model = Model('gpt2')

with model.generate(device_map='cuda', max_new_tokens=1) as generator:
    with generator.invoke('The Eiffel Tower is in the city of') as invoker:

        hidden_states = model.transformer.h[-1].output[0].t(0).save()

output = generator.output
hidden_states = hidden_states.value
```

---

###### Cross Prompt Intervention


Intervention operations work cross prompt! Use two invocations within the same generation block and operations can work between them.

Here 

---

###### Running Remotely


Running the engine API remotely on LLaMA 65b and saving the hidden states of the last layer:

```python
from engine import Model

model = Model('decapoda-research/llama-65b-hf')
with model.generate(device_map='server', max_new_tokens=1) as generator:
    with generator.invoke('The Eiffel Tower is in the city of') as invoker:

        hidden_states = model.model.layers[-1].output[0].save()

output = generator.output
hidden_states = hidden_states.value
```

More examples can be found in `engine/examples/`

## Inference Server

Source for the NDIF server is found in the `server/` directory.

- Edit `server/config.yaml` for your requirements. 
    - `PORT` : Flask port
    - `RESPONSE_PATH` : Where to store disk offloaded response data

#### Installation

Clone this repository and create the `ndif` conda environment:

```bash
cd ndif
conda env create -f server/environment.yaml
```

Start the server with:

```bash
python -m server
``` 
