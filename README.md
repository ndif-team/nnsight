# NDIF Repository
## engine API

The `engine/` directory contains the engine package for interpreting and manipulating the internals of large language models.

- `engine/model_checkpoints` is set to be the default huggingface hub cache directory. Contains by default models found on the NDIF server with their respective configurations.

#### Installation

Install this package through pip by running:

`pip install git+https://github.com/JadenFiotto-Kaufman/ndif`

#### Examples

```python
from engine import Model

model = Model('gpt2')

with model.invoke('The Eiffel Tower is in the city of') as invoker:

    hidden_states = model.transformer.h[-1].output[0]

output = model(device_map='cuda', max_new_tokens=1)

```

```python
<insert server example>
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
