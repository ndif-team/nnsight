from typing import Dict, Optional

import torch

from ..base import NNsight


class LoadableMixin(NNsight):

    def __init__(self, *args, rename: Optional[Dict[str,str]] = None,**kwargs) -> None:

        if not isinstance(args[0], torch.nn.Module):

            model = self._load(*args, **kwargs)

        else:

            model = args[0]
            
        super().__init__(model, rename=rename)

    def _load(self, *args, **kwargs) -> torch.nn.Module:

        raise NotImplementedError()
