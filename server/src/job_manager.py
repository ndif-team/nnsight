
import datetime
import logging
from multiprocessing import Process, Queue

from src.jobstatus import JobStatus
from baukit import nethook
from src.model_loader import ModelLoader
from utils import model_utils


class JobManager(Process):
    def __init__(
            self,
            model_name:str,
            job_queue:Queue, 
            results_dict:dict,
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
            "num_layers": ml.num_layers,
            "layer_name": ml.layer_name_format,
            "mlp_module_name_format": ml.mlp_module_name_format,
            "attn_module_name_format": ml.attn_module_name_format
        })

        self.model = ml.model
        self.tokenizer = ml.tokenizer

        while True:

            request = self.job_queue.get()
            self.submit(request)


    def process(self, request):

        prompts = request["prompt"]

        job_result = []

        for cur_propmt in prompts:
            txt, ret_dict = model_utils.generate_fast(
                self.model, self.tokenizer,
                [cur_propmt], max_new_tokens=request["max_new_tokens"],
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

                tokenized = self.tokenizer([cur_propmt], return_tensors="pt", padding = True).to(
                    next(self.model.parameters()).device
                )
                with nethook.TraceDict(
                    self.model,
                    layers = requested_modules,
                ) as traces:
                    outputs = self.model(**tokenized)

                for module in requested_modules:
                    result["activations"][module] = model_utils.untuple(traces[module].output).cpu().numpy().tolist()
                
            job_result.append(result)

        return job_result

    def submit(self, request):
        job_id = request['job_id']

        self.results_dict[job_id] = {
            'status' : JobStatus.SUBMITTED.name,
            'timestamp' : str(datetime.datetime.now()),
            'description' : 'Your job has been submitted and is running!'
        }
        
        logging.info(f"Job ID '{job_id}' running")

        try: 

            job_result = self.process(request)

        except Exception as exception:

            self.results_dict[job_id] = {
                'status' : JobStatus.ERROR.name,
                'timestamp' : str(datetime.datetime.now()),
                'description' : 'Your job errored out'
            }

            logging.error(str(exception))



        self.results_dict[job_id] = {
            'status' : JobStatus.COMPLETED.name,
            'timestamp' : str(datetime.datetime.now()),
            'description' : 'Your job has been completed',
            'data' : job_result
        }

        logging.info(f"Job ID '{job_id}' completed")


            




