from __future__ import annotations

from typing import Union

import torch
import uuid

class Promise(list):

    execution_graph = list()
    promises = dict()

    @classmethod
    def compile(cls):
        return list(reversed(Promise.execution_graph)), {id: promise.to_dict() for id, promise in Promise.promises.items()}

    @classmethod
    def clear(cls):
        Promise.promises.clear()
        Promise.execution_graph.clear()

    def __init__(self, args, shape, command='GET') -> None:

        self._value = None
        self._shape = shape
        self._command = command
        self._args = args
        self._id = str(uuid.uuid4())

        Promise.promises[self._id] = self
            
    def __repr__(self) -> str:
        return f"{self._command}({','.join([str(arg) for arg in self._args])})" if self._value is None else str(self.value)
    
    def __getitem__(self, key):

        output = torch.zeros(self._shape, device='meta')[key]

        return Promise([self, key], output.shape, command='SLC')
    
    def __add__(self: Promise, other: Union[Promise, torch.Tensor]):

        output = torch.zeros(self.shape, device='meta') + torch.zeros(other.shape, device='meta')

        model_state = Promise([self, other], output.shape, command='ADD')

        return model_state
    
    def copy(self):

        promise = Promise([self], self.shape, command='CPY')
        promise.execute()

        return promise

    def execute(self):
        Promise.execution_graph.append(self._id)

    def to_dict(self):
        return {'command' : self._command, 'id': self._id, 'args' : [arg._id if isinstance(arg, Promise) else arg for arg in self._args]}

    @property
    def shape(self):
        return self._shape
    
    @property
    def value(self):
        return self._value if self._value is not None else str(self)
    
    @value.setter
    def value(self, value): 
        self._value = value 
    