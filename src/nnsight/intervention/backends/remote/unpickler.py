from __future__ import annotations

from pickle import BUILD, _Unpickler

from dill import Unpickler


class RemoteUnpickler(Unpickler):
    pass
    
