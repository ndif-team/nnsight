from typing import Union, List

class Response:

    def __init__(self,
        activations,
        answer,
        generated_text,
        ) -> None:
        
        
        self.activations = activations
        self.answer = answer
        self.generated_text = generated_text
