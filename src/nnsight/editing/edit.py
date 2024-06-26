from typing import List

import torch

from ..envoy import Envoy

class Edit():
    """The Edit class represents an operation to add an attachment 
    (submodule) to a provided PyTorch module.

    Attributes:
        base (Envoy): The base is the module to which the attachment is added. 
                      This is typically an instance of the Envoy class.
        name (str): The name is the attribute name under which the attachment 
                    will be accessible in the base module.
        attachment (torch.nn.Module): A PyTorch module to be added to the base module.

    Methods:
        edit(): Adds the attachment to the base module under the specified name.
    """

    def __init__(
        self,
        base: Envoy,
        name: str,
        attachment: torch.nn.Module
    ):
        """Initializes an Edit instance with the base module, the name for the attachment,
        and the attachment itself.

        Args:
            base (Envoy): The base module to which the attachment will be added.
            name (str): The name under which the attachment will be accessible.
            attachment (torch.nn.Module): The PyTorch module to attach to the base.
        """
        self.base = base
        self.name = name
        self.attachment = attachment

    def edit(self):
        """Adds the attachment to the base module and sets it as an 
        attribute with the specified name.
        """
        self.base._add_envoy(
            self.attachment,
            self.name
        )
        
        setattr(
            self.base,
            self.name,
            self.attachment
        )

    def __repr__(self) -> str:
        return f"Edit(base={self.base}, name={self.name}, attachment={self.attachment})"

def apply_edits(edits: List[Edit]):
    """Applies a list of edits, adding each attachment to its respective base module.

    Args:
        edits (List[Edit]): A list of Edit instances to be applied.
    """
    for edit in edits:
        edit.edit()