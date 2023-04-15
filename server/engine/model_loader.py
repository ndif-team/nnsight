import torch
from baukit import nethook
#from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
import warnings
import time

class ModelLoader:
    def __init__(self, MODEL_NAME, dtype = torch.float16, device_map = "balanced") -> None:
        self.MODEL_NAME = MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) 
       
        #start_time = time.process_time_ns()
        #self.model = AutoModelForCausalLM.from_pretrained(
        #    MODEL_NAME, low_cpu_mem_usage=True, torch_dtype=dtype,
        #    device_map = device_map
        #)
        #print(f"Original load: {time.process_time_ns() - start_time} ns")

        ## Handled by accelerate:
        #self.model.eval()
        #if(device_map is None):
        #    self.model.cuda()
        
        ## ACCELERATE LOAD

        config = AutoConfig.from_pretrained(MODEL_NAME)

        with init_empty_weights():
           model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
        
        self.model = model
        self.extract_relavent_fields_from_config()

        # must tie weights before loading
        model.tie_weights()
        
        start_time = time.process_time_ns()
        self.model = load_checkpoint_and_dispatch(
                model, MODEL_NAME, device_map='auto',
            no_split_module_classes = self.no_split_module_classes
        )
        print(f"Load time: {time.process_time_ns()-start_time} ns") 

        nethook.set_requires_grad(False, self.model)


        for n, p in self.model.named_parameters():
            print(n, p.shape, p.device)

        if(self.model_type in ["gpt2", "gpt_neox", "llama"]):
            self.tokenizer.pad_token = self.tokenizer.eos_token            
        elif(self.model_type in [ "galactica"]):
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
            no_split_module_classes = ["GPT2Block"]
        elif(hasattr(self.model, "gpt_neox")):
            model_type = "gpt-neox"
            no_split_module_classes = ["GPTNeoXLayer"]
        elif("llama" in config._name_or_path):
            model_type = "llama"
            no_split_module_classes = ["LlamaDecoderLayer"]
        elif("galactica" in config._name_or_path):
            model_type = "galactica"
            no_split_module_classes  = ["OPTDecoderLayer"]
        else:
            warnings.warn("unknown model type >> unable to extract relavent fields from config")

        self.n_layer = None
        self.n_embd = None
        self.n_attn_head = None
        self.max_seq_length = None

        self.layer_name_format = None
        self.layer_names = None
        self.mlp_module_name_format = None
        self.attn_module_name_format = None
        self.ln_f_name = None
        self.unembedder_name = None
        self.embedder_name = None
        
        self.model_type = model_type
        self.no_split_module_classes = no_split_module_classes

        if(model_type in ["llama", "galactica"]):
            self.n_layer = config.num_hidden_layers
            self.n_embd = config.hidden_size
            self.n_attn_head = config.num_attention_heads
            self.max_seq_length = config.max_sequence_length

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
            self.n_layer = config.n_layer
            self.n_embd = config.n_embd
            self.n_attn_head = config.n_head
            self.max_seq_length = config.n_ctx

            self.layer_name_format = "transformer.h.{}"
            self.embedder_name = "transformer.wte"
            self.ln_f_name = "transformer.ln_f"
            self.unembedder_name = "lm_head"
            self.mlp_module_name_format = "transformer.h.{}.mlp"
            self.attn_module_name_format = "transformer.h.{}.attn"
    
        # print("num_layers >> ", self.num_layers)
        if(model_type is not None):
            self.layer_names = [self.layer_name_format.format(i) for i in range(self.n_layer)]
