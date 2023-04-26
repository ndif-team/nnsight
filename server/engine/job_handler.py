
import logging
from multiprocessing import Process, Queue

from engine.model_manager import ModelManager
from engine.models.submit import Request
from engine.models.result import JobStatus, Result
from engine.results_dict import ResultsDict


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

        model_manager = ModelManager(MODEL_PATH=self.model_name)
        self.model_manager = model_manager

        self.info_dict.update({
            "n_layer": model_manager.n_layer,
            "n_embd" : model_manager.n_embd,
            "n_attn_head" : model_manager.n_attn_head,
            "max_seq_length": model_manager.max_seq_length,
            
            "layer_name_format": model_manager.layer_name_format,
            "mlp_module_name_format": model_manager.mlp_module_name_format,
            "attn_module_name_format": model_manager.attn_module_name_format
        })


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
        
        print(request)

        prompts = request.prompts

        response = []

        # ! Can't run multiple prompts due to limited support for newer models like `galactica` and `llama` on huggingface accelerate
        for cur_propmt in prompts:
            txt, ret_dict = self.model_manager.generate(
                [cur_propmt], 
                max_out_len=request.max_out_len,
                argmax_greedy=request.generate_greedy,
                top_k = request.top_k,
                request_activations=request.activation_requests.layers
            )  

            # ! technically `generate` can accept a batch of prompts and return a batch of results
            result = {
                "generated_text": txt[0],
                "input_tokenized": ret_dict["input_tokenized"][0],
                "generated_tokens": ret_dict["generated_tokens"][0],
            }
            if("activations" in ret_dict):
                result["activations"] = {
                    k: ret_dict["activations"][k][0].tolist() for k in ret_dict["activations"]
                }
            response.append(result)
        
        return response

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
        

            




