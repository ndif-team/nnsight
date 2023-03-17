import torch
import transformers
from tqdm import tqdm
import transformers
from utils import model_utils
from baukit import nethook
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

class ModelLoader:
    def __init__(self, MODEL_NAME) -> None:
        self.MODEL_NAME = MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, low_cpu_mem_usage=True, #torch_dtype=torch_dtype,
            # device_map = "balanced"
        )
        self.model.eval().cuda()

        # for n, p in self.model.named_parameters():
        #     print(n, p.shape, p.device)

        nethook.set_requires_grad(False, self.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"{MODEL_NAME} ==> device: {self.model.device}, memory: {self.model.get_memory_footprint()}")

        self.layer_names = [
            n
            for n, m in self.model.named_modules()
            if (re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)
