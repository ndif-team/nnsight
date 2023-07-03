import torch

torch.set_default_device('meta')

class NDIFModule(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:

        self.input_shape = None
        self.output_shape = None

        super().__init__(*args, **kwargs)

        self.register_forward_hook(NDIFModule.hook)

    @staticmethod
    def hook(module, input, output):

        module.input_shape = input.shape if isinstance(input, torch.Tensor) else [_input.shape for _input in input]
        module.output_shape = output.shape if isinstance(output, torch.Tensor) else [_output.shape for _output in output]

    def activations(self):

        pass

torch.nn.Module = NDIFModule

from transformers import GPT2Config, GPT2Model, GPT2Tokenizer



# Initializing a GPT2 configuration
configuration = GPT2Config()

# Initializing a model (with random weights) from the configuration
model = GPT2Model(configuration)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

input_ids = tokenizer("Hello world", return_tensors='pt')["input_ids"]

output = model(input_ids)


breakpoint()