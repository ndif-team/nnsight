from __future__ import annotations

from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Union
from typing_extensions import override
from ..util import Value
import torch

INTERVENTIONS = {}

# MAYBE DO CALL ON VALUE SET

class Intervention:
    '''
    An Intervention represents an action that needs to be carried out
    during the execution of a Model, and a store of value for those actions.
    
    Class Attributes
    ----------
    interventions : Dict[str, Intervention]
        stores a mapping between an Intervention's unique id and the Intervention.

    Attributes
    ----------
    _value : str
        value store of Intervention
    id : str
        unique id of Intervention
    listeners : Dict[str, Intervention]
        mapping from id to Intervention for parent Interventions that depend on
        this Interventions
    '''

    interventions:Dict[str, Intervention] = OrderedDict()

    @classmethod
    def clear(self) -> None:
        '''Clears the Intervention class attribute mapping'''
        Intervention.interventions.clear()

    @classmethod
    def parse(cls, arg:Value, promises:Dict[str, Dict]) -> Union[Intervention,Value]:
        '''
        Parses a promise and it's args into Interventions
        '''
        if isinstance(arg, str) and arg in promises:

            promise = promises[arg]
            promise['args'] = [Intervention.parse(arg, promises) for arg in promise['args']]
            promise = Intervention.create(**promise)

            return promise
        return arg

    @classmethod
    def from_execution_graph(cls, execution_graph:List[str], promises:Dict[str, Dict]) -> None:
        '''
        Parses the information from Promises into Interventions.
        
        Parameters
        ----------
            execution_graph : List[str]
                List of ids of Promises to be executed in order of execution. Returned from Promise.compile()
            promises : Dict
                Mapping of Promise id to Dict of attributes that make up a promise: 'args', 'command', and 'id'. Returned from Promise.compile()
        '''

        listener = None

        for id in execution_graph:

            intervention = Intervention.parse(id, promises)
            intervention.listen(listener)
            listener = intervention

    def __init__(self, id:str) -> None:

        self._value = None
        self.id = id
        self.listeners = {}

        Intervention.interventions[self.id] = self

    @classmethod
    def create(cls, args:List[Union[Intervention,Value]], id:str, command:str) -> Intervention:
        '''
        If an Intervention with the given id already exists, return it.
        Otherwise create a new Intervention with subtype depending on command.

        Parameters
        ----------
            args : List[Union[Intervention,Value]]
                List of arguments for Intervention. Can be anything
            id : str
                id of Intervention
            command : str
                String denoting what kind of Intervention
        '''

        if id in Intervention.interventions:

            return Intervention.interventions[id]
                
        return INTERVENTIONS[command](*args,id)
    
    @abstractmethod
    def __call__(self) -> None:
        '''
        Abstract method that attempts to signal listners that it's value is changed.
        Inheritors should perform Intervention subtype specific actions then call super().__call__()
        if value is updated.
        '''

        print(self)
        
        for listener in list(self.listeners.values()):

            listener()

    def listen(self, listener:Intervention) -> None:
        '''
        Adds listener to listeners

        Parameters
        ----------
            listener : Intervention
                listener to add to listeners
        '''

        if listener is not None:

            self.listeners[listener.id] = listener

    def destroy(self) -> None:
        '''
        Removes reference to self from id to Intervention mapping and sets its self._value 
        to None.
        '''

        print(f"Destroying {str(self)}")

        del Intervention.interventions[self.id]

        self._value = None

    def stop_listening(self, id:str) -> None:
        '''
        Removes Intervention with id from listeners. If there exist no more listeners,
        destory self.

        Parameters
        ----------
            id : str
                id of Intervention to remove from listeners
        '''

        del self.listeners[id]

        if len(self.listeners) == 0:

            self.destroy()

    def get_value(self, listener_id:str) -> torch.Tensor:
        '''
        Gets the Intervention's value. Requires an Intervention id in listeners.
        Removes the listener with id listener_id from listeners.
        If value is None, raise ValueError.

        Parameters
        ----------
            listener_id : str
                id of Intervention that requests value
        '''
        if listener_id not in self.listeners:

            return None

        if self._value is None:

            raise ValueError(f"Listener '{str(Intervention.interventions[listener_id])}' referenced value '{str(self)}' before assignment")

        value = self._value

        self.stop_listening(listener_id)

        return value
    
    def set_value(self, value:torch.Tensor, listener_id:str) -> None:
        '''
        Sets the Intervention's value. Requires an Intervention id in listeners.
        Removes the listener with id listener_id from listeners.

        Parameters
        ----------
            value : torch.Tensor
                value to set
            listener_id : str
                id of Intervention that requests to set value
        '''
        if listener_id not in self.listeners:

            return None
        
        self.stop_listening(listener_id)

        self._value = value

class Add(Intervention):

    def __init__(self, arg1:Intervention, arg2:Intervention, *args, **kwargs) -> None:

        super().__init__(*args,**kwargs)

        self.arg1 = arg1
        self.arg2 = arg2
        self.arg1.listen(self)
        self.arg2.listen(self)

    def __repr__(self) -> str:
        return f"ADD({str(self.arg1)},{self.arg2})"

    def __call__(self):

        if self.arg1._value is not None and self.arg2._value is not None:

            self._value = self.arg1.get_value(self.id) + self.arg2.get_value(self.id)

            super().__call__()

class Get(Intervention):

    gets:Dict[str, List[Get]] = dict()

    @classmethod
    def layers(cls):

        return [layer.replace('.input', '').replace('.output', '') for layer in list(Get.gets.keys())]
    
    def __init__(self, module_name:str, prompt_index:int, *args, **kwargs) -> None:

        super().__init__(*args,**kwargs)

        self.module_name:str = module_name
        self.prompt_index:int = prompt_index

        if module_name in Get.gets:

            Get.gets[module_name].append(self)

        else:

            Get.gets[module_name] = [self]

    def __repr__(self) -> str:
        return f"GET({self.module_name}[{self.prompt_index}])"

    def init_value(self, value): 

        if self._value is not None:

            raise ValueError(f"Get '{str(self)}' already initialized")

        self._value = value[self.prompt_index]

        self()

        if self._value is not None:

            value[self.prompt_index] = self._value

class Set(Intervention):

    def __init__(self, arg1:Intervention, arg2:Intervention, *args, **kwargs) -> None:
        
        super().__init__(*args,**kwargs)

        self.arg1 = arg1
        self.arg2 = arg2
        self.arg1.listen(self)
        self.arg2.listen(self)

    def __repr__(self) -> str:
        return f"SET({str(self.arg1)},{self.arg2})"

    def __call__(self) -> Any:

        if self.arg1._value is not None:
    
            self.arg1.set_value(self.arg2.get_value(self.id), self.id)

            super().__call__()

            self.destroy()
        
class Copy(Intervention):

    def __init__(self, arg1:Intervention, *args, **kwargs) -> None:

        super().__init__(*args,**kwargs)

        self.arg1 = arg1
        self.arg1.listen(self)

    def __repr__(self) -> str:
        return f"COPY({str(self.arg1)})"

    def __call__(self):

        if self.arg1._value is not None:

            self._value = self.arg1.get_value(self.id)

            super().__call__()

    @override
    def destroy(self) -> None:
        pass

class Slice(Intervention):

    def __init__(self, arg1:Intervention, slice, *args, **kwargs) -> None:

        super().__init__(*args,**kwargs)

        self.arg1 = arg1
        self.arg1.listen(self)
        self.slice = slice

    def __call__(self):

        self._value = self.arg1.get_value(self.id)[self.slice]

        super().__call__()

class Tensor(Intervention):

    @classmethod
    def to(cls, device):
        for intervention in Intervention.interventions.values():
            if isinstance(intervention, Tensor):
                intervention._value = intervention._value.to(device)

    def __init__(self, value:torch.Tensor, *args, **kwargs) -> None:

        super().__init__(*args,**kwargs)

        self._value = value

    def __repr__(self) -> str:
        return f"TENSOR({self.id})"

INTERVENTIONS.update({'GET': Get, 'SET': Set, 'CPY': Copy, 'ADD': Add, 'TNS': Tensor})

def intervene(activations, module_name):

    if module_name in Get.gets:

        for get in Get.gets[module_name]:

            get.init_value(activations)

    return activations


def output_intervene(activations, module_name):

    module_name = f"{module_name}.output"

    return intervene(activations, module_name)


def input_intervene(activations, module_name):

    module_name = f"{module_name}.input"

    return intervene(activations,module_name)
