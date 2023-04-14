
import logging
from multiprocessing import Process, Queue

import torch
from baukit import nethook
from engine.model_loader import ModelLoader
from engine.models.submit import Request
from engine.models.result import JobStatus, Result
from engine.results_dict import ResultsDict
from engine.utils import model_utils


class JobManager(Process):
    def __init__(
            self,
            model_name:str,
            job_queue:Queue, 
            results_dict:ResultsDict,
            info_dict:dict,
        ):
        self.model_name = model_name
        self.job_queue = job_queue
        self.results_dict = results_dict
        self.info_dict = info_dict

        super().__init__()

    def run(self):

        ml = ModelLoader(MODEL_NAME=self.model_name)

        self.info_dict.update({
            "n_layer": ml.n_layer,
            "n_embd" : ml.n_embd,
            "n_attn_head" : ml.n_attn_head,
            "max_seq_length": ml.max_seq_length,
            
            "layer_name_format": ml.layer_name_format,
            "mlp_module_name_format": ml.mlp_module_name_format,
            "attn_module_name_format": ml.attn_module_name_format
        })

        self.model = ml.model
        self.tokenizer = ml.tokenizer

        while True:

            try:
                # TODO: BATCHING! currently gets one request each time
                request = self.job_queue.get()
                self.submit(request)

            except Exception as exception:

                job_id = request.job_id

                self.results_dict[job_id] = Result(
                    job_id = job_id,
                    status = JobStatus.ERROR,
                    description = "Your job errored out"
                )

                logging.exception("Exception occured in job processing")

    def process(self, request:Request):

        prompts = request.prompt

        job_result = []

        # ! Can't run multiple prompts due to limited support for newer models like `galactica` and `llama` on huggingface accelerate
        for cur_propmt in prompts:
            txt, ret_dict = model_utils.generate_fast(
                self.model, self.tokenizer,
                [cur_propmt], max_new_tokens=request.max_new_tokens,
                argmax_greedy=request.generate_greedy,
                top_k = request.top_k,
                get_answer_tokens = True,
            )  

            result = {
                "generated_text": txt,
                "generated_tokens": ret_dict["generated_tokens"],
                "answer": ret_dict["answer"],
            }

            if request.activation_requests is not None:
                result["activations"] = {}
                requested_modules = request.activation_requests[0].layers

                tokenized = self.tokenizer([cur_propmt], return_tensors="pt", padding = True).to(next(self.model.parameters()).device)
                result["input_tokenized"] = [(self.tokenizer.decode(t), t.item()) for t in tokenized.input_ids[0]]
                # print(result)

                with nethook.TraceDict(
                    self.model,
                    layers = requested_modules,
                ) as traces:  
                    outputs = self.model(**tokenized)

                for module in requested_modules:
                    result["activations"][module] = model_utils.untuple(traces[module].output).cpu().numpy().tolist()
                
                # clear up the precious GPU memory as soon as the inference is done
                del(traces)
                del(outputs)
                torch.cuda.empty_cache()
                
            job_result.append(result)

        return job_result

    def submit(self, request:Request):
        
        job_id = request.job_id

        self.results_dict[job_id] = Result(
            job_id = job_id,
            status = JobStatus.SUBMITTED,
            description = "Your job has been submitted and is running!"
        )
        
        job_result = self.process(request)

        self.results_dict[job_id] = Result(
                    job_id = job_id,
                    status = JobStatus.COMPLETED,
                    description = "Your job has been completed",
                    data = job_result,
                )
        

            




