from .gpt2 import gpt2_attr_map

model_class_mappings = {
    "GPT2LMHeadModel" : gpt2_attr_map
}

def get_attr_map(model_name):
    if model_name in model_class_mappings:
        return model_class_mappings[model_name]
    else:
        raise ValueError(f"Model {model_name} not supported.")
