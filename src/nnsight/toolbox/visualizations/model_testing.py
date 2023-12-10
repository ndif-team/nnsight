from nnsight import LanguageModel
import torch as t
from rich import print
from typing import Union
from rich.console import Console
from rich.table import Table

def test_prompt(
        model: LanguageModel,
        prompt: str,
        answer: Union[str, int] = ""):
    """test_prompt returns a table of the top 10 tokens and their probabilities
    
    Args:
        model (LanguageModel): The model to test
        prompt (str): The prompt to pass through the model
        answer (Union[str, int], optional): The token to test. Can take a string or token value. Defaults to "".

    """

    # Pass prompt through model
    with model.invoke(prompt) as invoker:
        pass
            
    # Get logits of output
    logits = invoker.output.logits
    # Get final token logits
    final_token_logits = logits[0,-1,:] 
    probs = final_token_logits.softmax(dim=-1)

    # Sort probabilities descending
    sorted_indices = t.argsort(probs, descending=True)
    # Tokenize answer if it is a string
    if type(answer) == str:
        answer = model.tokenizer(answer).input_ids[0]
    
    # Initialize table
    table = Table(title="Top K Tokens")

    table.add_column("K", justify="center", style="cyan")
    table.add_column("Logit", justify="center", style="cyan")
    table.add_column("Prob", justify="center", style="cyan")
    table.add_column("Token", justify="left", style="cyan")

    # Get probability at answer token 
    rank = (sorted_indices == answer).nonzero(as_tuple=True)[0]
    
    table.add_row(f"{rank.item()}", f"{final_token_logits[answer]:.3f}", f"{probs[answer]:.3f}", f"|{model.tokenizer.decode(answer)}", style="bold green")

    # Add top 10 tokens to table
    for i in range(10):
        tok = sorted_indices[i]
        prob = probs[tok]
        table.add_row(f"{i}", f"{final_token_logits[tok]:.3f}", f"{prob:.3f}", f"|{model.tokenizer.decode(tok)}")

    console = Console()
    console.print(table)
    