from .gpt2 import GPT2AttentionAltered

model_alterations = {
    "GPT2LMHeadModel" : GPT2AttentionAltered
}

def get_alteration(model_name):
    if model_name in model_alterations:
        return model_alterations[model_name]
    else:
        return {}