import einops 

gpt2 = {
    "wte" : "embed",
    "wpe" : "pos_embed",
    "drop" : "dropout",
    "h" : "layers" ,
    "ln_f" : "ln_final",
    "lm_head" : "unembed"
}

custom_modules = {
    "c_attn" : None
}

# def refactor_attention(attention):
#     einops.rearrange(attention, 'b h t d -> b t h d')