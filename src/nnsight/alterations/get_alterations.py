from .gpt2 import gpt2_attr_map, GPT2Patcher

# Ideally, model_alterations keys would be class names.
# Using repo names because we need to load a patcher before loading model weights.
model_alterations = {
    "openai-community/gpt2" : (gpt2_attr_map, GPT2Patcher)
}

def get_attr_map(model_name):
    if model_name in model_alterations:
        return model_alterations[model_name][0]
    else:
        raise ValueError(f"Attribute mappings for {model_name} not supported.")
    
def get_alteration(model_name):
    if model_name in model_alterations:
        return model_alterations[model_name][1]
    else:
        raise ValueError(f"Model alterations for {model_name} not supported.")