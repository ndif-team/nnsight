import accelerate
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    GenerationMixin,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import torch
#from torch.fx
from .Graph import Graph
#from transformers.models.gpt2
model_name_or_path = 'gpt2'
with accelerate.init_empty_weights(include_buffers=True):
    config = AutoConfig.from_pretrained(
        model_name_or_path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        config=config,
        padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    model: PreTrainedModel = AutoModelForCausalLM.from_config(
        config
    )

graph = Graph.trace(model.transformer.h[0], torch.empty((1,10,768), device='meta'))

breakpoint()