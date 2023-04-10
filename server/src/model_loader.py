import torch
import transformers
from tqdm import tqdm
import transformers
from utils import model_utils
from baukit import nethook
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import warnings

class ModelLoader:
    def __init__(self, MODEL_NAME, dtype = torch.float16, device_map = "balanced") -> None:
        self.MODEL_NAME = MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) 
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, low_cpu_mem_usage=True, torch_dtype=dtype,
            device_map = device_map
        )
        self.model.eval()
        if(device_map is None):
            self.model.cuda()
        nethook.set_requires_grad(False, self.model)

        self.extract_relavent_fields_from_config()

        for n, p in self.model.named_parameters():
            print(n, p.shape, p.device)

        if(self.model_type in ["gpt2", "gpt_neox"]):
            self.tokenizer.pad_token = self.tokenizer.eos_token            
        elif(self.model_type in ["llama", "galactica"]):
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
            
        print(f"{MODEL_NAME} ==> device: {self.model.device}, memory: {self.model.get_memory_footprint()}")    

    # tested for GPT-j, galactica and LLaMa
    def extract_relavent_fields_from_config(self):
        """
        extracts a bunch of highly used fields from different model configurations
        """
        config = self.model.config
        self.vocab_size = config.vocab_size

        model_type = None
        if(hasattr(self.model, "transformer")):
            model_type = "gpt2"
        elif(hasattr(self.model, "gpt_neox")):
            model_type = "gpt-neox"
        elif("llama" in config._name_or_path):
            model_type = "llama"
        elif("galactica" in config._name_or_path):
            model_type = "galactica"
        else:
            warnings.warn("unknown model type >> unable to extract relavent fields from config")

        self.num_layers = None
        self.layer_name_format = None
        self.layer_names = None
        self.mlp_module_name_format = None
        self.attn_module_name_format = None
        self.ln_f_name = None
        self.unembedder_name = None
        self.embedder_name = None
        
        self.model_type = model_type


        if(model_type in ["llama", "galactica"]):
            self.num_layers = config.num_hidden_layers
            layer_name_prefix = "model"
            if(model_type == "galactica"):
                layer_name_prefix = "model.decoder"
            
            self.layer_name_format = layer_name_prefix + ".layers.{}"

            self.embedder_name = "model.embed_tokens"
            self.ln_f_name = "model.norm" if model_type=="llama" else "model.decoder.final_layer_norm"
            self.unembedder_name = "lm_head"

            if(model_type == "llama"):
                self.mlp_module_name_format = "model.layers.{}.mlp"
            else:
                self.mlp_module_name_format = "model.layers.{}.fc2" # this is the output of mlp in galactica. the input is on model.layers.{}.fc1
            self.attn_module_name_format = "model.layers.{}.self_attn"

        elif(model_type in ["gpt2", "gpt-neox"]):
            self.num_layers = config.n_layer
            self.layer_name_format = "transformer.h.{}"

            self.embedder_name = "transformer.wte"
            self.ln_f_name = "transformer.ln_f"
            self.unembedder_name = "lm_head"

            self.mlp_module_name_format = "transformer.h.{}.mlp"
            self.attn_module_name_format = "transformer.h.{}.attn"
    
        # print("num_layers >> ", self.num_layers)
        if(model_type is not None):
            self.layer_names = [self.layer_name_format.format(i) for i in range(self.num_layers)]
