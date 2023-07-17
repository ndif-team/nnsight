from __future__ import annotations

from typing import Dict, List, Tuple, Union

import torch
import uuid
from .util import Value

class Promise(list):
    '''
    A Promise represents an action that is expected to be carried out when running a Model.

    Class Attributes
    ----------
    execution_graph : List[str]
        list of ids of promises that need to be executed in reverse order. These should only
        be ids of Copy and Set Promises as only they actually effect Model inference.
    promises : Dict[str,Promise]
        Mapping of id to Promise to de-reference ids and to find and update Promises
        after Model execution.

    Attributes
    ----------
    _input : Promise
        Promise encapsulating the value of the Module's input. None before referencing
    _output : Promise
        Promise encapsulating the value of the Module's output. None before referencing
    output_shape : torch.Size
        shape of Module output
    input_shape : torch.Size
        shape of Module input
    module_path : str
        path of Module in Model tree
    '''

    execution_graph:List[str] = list()
    promises:Dict[str,Promise] = dict()

    @classmethod
    def compile(cls) -> Tuple[List[str],Dict[str,Dict]]:
        '''
        Class method to return necessary information for parsing into Interventions.

        Returns
        ----------
        execution_graph : List[str]
            execution graph of ids in execution order.
        promises : Dict[str,Dict]
            Mapping of id to Dict where Dict are the keys and values needed to build an Intervention.

        '''
        return list(reversed(Promise.execution_graph)), {id: promise.to_dict() for id, promise in Promise.promises.items()}

    @classmethod
    def clear(cls) -> None:
        '''
        Class method to clear class attributes after completed run.
        '''
        Promise.promises.clear()
        Promise.execution_graph.clear()

    @classmethod 
    def wrap(cls, value:Union[Promise,Value]):

        if isinstance(value, Promise):
            return value
        if not isinstance(value, torch.Tensor):
            value = torch.Tensor([value])

        return Promise([value], value.shape, command='TNS')

    @classmethod
    def update_prompt_index(self, prompt_index:int) -> None:

        for promise in Promise.promises.values():
            if promise.command == 'GET':
                promise.args.append(prompt_index)

    def __init__(self, args, shape, command='GET') -> None:

        self._value = None
        self._shape = shape
        self.command = command
        self.args = args
        self.id = str(uuid.uuid4())

        Promise.promises[self.id] = self
            
    def __repr__(self) -> str:
        return f"{self.command}({','.join([str(arg) for arg in self.args])})" if self._value is None else str(self.value)
    
    def __getitem__(self, key) -> Promise:
        '''
        Overridden method that creates a Promise to slice/access values from another Promise
        
        Parameters
        ----------
        key : 
            key or slice

        Returns
        ----------
        promise : Promise
            a Slice Promise
        '''
        output = torch.zeros(self._shape, device='meta')[key]

        return Promise([self, key], output.shape, command='SLC')
    
    def __add__(self, other: Union[Promise,Value]) -> Promise:
        '''
        Overridden method that creates a Promise to add two Promises.
        
        Parameters
        ----------
        other : Union[Promise,Value]
            second argument to add. If not a promise, wrap the Value in a Promise

        Returns
        ----------
        promise : Promise
            an Add Promise
        '''

        other = Promise.wrap(other)

        output = torch.zeros(self.shape, device='meta') + torch.zeros(other.shape, device='meta')

        model_state = Promise([self, other], output.shape, command='ADD')

        return model_state
    
    def copy(self):

        promise = Promise([self], self.shape, command='CPY')
        promise.execute()

        return promise

    def execute(self):
        Promise.execution_graph.append(self.id)

    def to_dict(self):
        return {'command' : self.command, 'id': self.id, 'args' : [arg.id if isinstance(arg, Promise) else arg for arg in self.args]}

    @property
    def shape(self):
        return self._shape
    
    @property
    def value(self):
        return self._value if self._value is not None else str(self)
    
    @value.setter
    def value(self, value): 
        self._value = value 
    