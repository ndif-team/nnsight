from __future__ import annotations

import uuid
from typing import Dict, List, Tuple, Union

import torch
from typing_extensions import override
import numpy as np
from .util import Value, apply

class Promise(list):
    '''
    A Promise represents an action that is expected to be carried out when running a Model.

    Class Attributes
    ----------
        execution_graph : List[str]
            list of ids of promises that need to be executed in order. These should only
            be ids of Copy and Set Promises as only they actually effect Model inference.
        promises : Dict[str,Promise]
            Mapping of id to Promise to de-reference ids and to find and update Promises
            after Model execution.

    Attributes
    ----------
        _value : torch.Tensor
            value tensor
        _shape : torch.Size
            size of value tensor if set
        command : str
            which command this Promise represents
        args : List[Union[Promise,Value]]
            list of arguments
        id : str
            unique uuid4 of Promise
    '''

    execution_graph: List[str] = list()
    promises: Dict[str, Promise] = dict()

    class Tokens(dict):

        tokens:Dict[str,int] = None

        def __init__(self, promise:Promise) -> None:
            
            self.promise = promise

        def __getitem__(self, key):

            if isinstance(key, str):
            
                return self.promise[:, Promise.Tokens.tokens[key]]
            
            if isinstance(key, int):

                return self.promise[:, key]

    @classmethod
    def set_tokens(cls, tokenized:List[str]) -> None:

        Promise.Tokens.tokens = dict(zip(tokenized, np.arange(start=0, stop=len(tokenized))))

    @classmethod
    def compile(cls) -> Tuple[List[str], Dict[str, Dict]]:
        '''
        Class method to return necessary information for parsing into Interventions.

        Returns
        ----------
            List[str]
                execution graph of ids in execution order.
            Dict[str,Dict]
                Mapping of id to Dict where Dict are the keys and values needed to build an Intervention.
        '''
        return list(Promise.execution_graph), {id: promise.to_dict() for id, promise in Promise.promises.items()}

    @classmethod
    def clear(cls) -> None:
        '''
        Class method to clear class attributes after completed run.
        '''
        Promise.promises.clear()
        Promise.execution_graph.clear()

    @classmethod
    def wrap(cls, value: Union[Promise, Value]) -> Promise:
        '''
        Wraps a Value in a Promise. If already a Promise, return it.
        If not a torch.Tensor, make it a tensor.
        '''
        if isinstance(value, Promise):
            return value
        if not isinstance(value, torch.Tensor):
            value = torch.Tensor([value])

        return Promise([value], value.shape, command='TNS')

    @classmethod
    def update_batch_index(self, batch_index: int) -> None:

        for promise in Promise.promises.values():
            if promise.command == 'GET':
                promise.args.append(batch_index)

    def __init__(self, args: List[Union[Promise, Value]], shape: torch.Size, command: str = 'GET') -> None:

        self._value = None
        self._shape = shape
        self.command = command
        self.args = args
        self.id = str(uuid.uuid4())

        Promise.promises[self.id] = self

    def get_meta(self):

        return apply(self.shape, lambda x : torch.zeros(x, device='meta'), torch.Size)

    @override
    def __repr__(self) -> str:
        return f"{self.command}({','.join([str(arg) for arg in self.args])})" if self._value is None else str(self.value)

    @override
    def __getitem__(self, key) -> Promise:
        '''
        Overridden method that creates a Promise to slice/access values from another Promise

        Parameters
        ----------
            key : 
                key or slice

        Returns
        ----------
            Promise
                a Slice Promise
        '''

        if isinstance(self._shape, torch.Size):
            shape = self.get_meta()[key].shape
        else:
            shape = self.shape[key]

        return Promise([self, key], shape, command='SLC')

    @override
    def __add__(self, other: Union[Promise, Value]) -> Promise:
        '''
        Overridden method that creates a Promise to add two Promises.

        Parameters
        ----------
        other : Union[Promise,Value]
            second argument to add. If not a promise, wrap the Value in a Promise

        Returns
        ----------
        Promise
            an Add Promise
        '''

        other = Promise.wrap(other)

        output = self.get_meta() + other.get_meta()

        model_state = Promise([self, other], output.shape, command='ADD')

        return model_state

    def copy(self) -> Promise:

        promise = Promise([self], self.shape, command='CPY')
        promise.execute()

        return promise

    def execute(self) -> None:
        Promise.execution_graph.append(self.id)

    def to_dict(self) -> Dict[str, Value]:
        return {'command': self.command, 'id': self.id, 'args': [arg.id if isinstance(arg, Promise) else arg for arg in self.args]}

    @property
    def token(self) -> Promise:
        return Promise.Tokens(self)
    
    @property
    def t(self) -> Promise:
        return self.token

    @property
    def shape(self) -> torch.Size:
        return self._shape

    @property
    def value(self) -> Union[torch.Tensor, str]:
        return self._value if self._value is not None else str(self)

    @value.setter
    def value(self, value) -> None:
        self._value = value
