# NDIF Repository
## engine API

The `engine/` directory contains the engine package for interpreting and manipulating the internals of large language models.

- `engine/model_checkpoints` is set to be the default huggingface hub cache directory. Contains by default models found on the NDIF server with their respective configurations.
- Calling the model ends up calling [AutoModelForCausalLM.generate(...)](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate) and **kwargs passed to the call will be passed to generate. Refer to the linked docs for reference.

#### Installation

Install this package through pip by running:

`pip install git+https://github.com/JadenFiotto-Kaufman/ndif`

#### Examples

Running the engine API locally on gpt2 and saving the hidden states of the last layer:

```python
from engine import Model

model = Model('gpt2')

with model.invoke('The Eiffel Tower is in the city of') as invoker:

    hidden_states = model.transformer.h[-1].output[0].copy()

output = model(device_map='cuda', max_new_tokens=1)

```

Running the engine API remotely on LLaMA 65b and saving the hidden states of the last layer:

```python
from engine import Model

model = Model('decapoda-research/llama-65b-hf')

with model.invoke('The Eiffel Tower is in the city of') as invoker:

    hidden_states = model.model.layers[-1].output[0].copy()

output = model(device_map='server', max_new_tokens=1)
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
