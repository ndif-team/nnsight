# Inside src/mappings/mapping.py

from .gpt2 import gpt2

model_class_mappings = {
    "GPT2LMHeadModel" : gpt2
}

def get_mapping(model_name):
    if model_name in model_class_mappings:
        return model_class_mappings[model_name]
    else:
        return {}
