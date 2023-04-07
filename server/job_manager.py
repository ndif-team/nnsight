import torch
import transformers
from tqdm import tqdm
import transformers
from utils import model_utils
from baukit import nethook
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import os
import json
import time

from request_handler import RequestHandler

class JobManager:
    def __init__(
            self, mt,
            request_handler, 
            save_path,
        ):
        self.model = mt.model
        self.tokenizer = mt.tokenizer
        self.request_handler = request_handler
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
    
    def run(self):
        batch = self.request_handler.get_new_batch()
        if(batch == None):
            print("empty job queue, ", type(self.model))
        else:
            job_id, request = batch
            prompt = request["prompt"]
            print("running job with batch ==> ", job_id)

            txt, ret_dict = model_utils.generate_fast(
                self.model, self.tokenizer,
                prompt, max_new_tokens=request["max_new_tokens"],
                argmax_greedy=request["generate_greedy"],
                top_k = request["top_k"],
                get_answer_tokens = True,
            )
            result = {
                "generated_text": txt,
                "answer": ret_dict["answer"]
            }

            if(request["activation_requests"] is not None):
                result["activations"] = {}
                requested_modules = request["activation_requests"]["layers"]

                tokenized = self.tokenizer(prompt, return_tensors="pt", padding = True).to(
                    next(self.model.parameters()).device
                )
                with nethook.TraceDict(
                    self.model,
                    layers = requested_modules,
                ) as traces:
                    outputs = self.model(**tokenized)

                for module in requested_modules:
                    result["activations"][module] = model_utils.untuple(traces[module].output).cpu().numpy().tolist()

            with open(f"{self.save_path}/{job_id}.json", "w") as f:
                json.dump(result, f)
            print("finished running current job ==> ", job_id)
            self.request_handler.processed.append(job_id)

        time.sleep(5)   # add a delay
        self.run()      # keep checking for jobs 


            




