from __future__ import annotations

import uuid
from abc import abstractclassmethod, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Union

import torch
from typing_extensions import override

from .util import Value, apply

INTERVENTIONS_TYPES = {}


class Intervention:
    '''
    An Intervention represents an action that needs to be carried out
    during the execution of a Model, and a store of value for those actions.

    Class Attributes
    ----------
        interventions : Dict[str, Intervention]
            stores a mapping between an Intervention's unique id and the Intervention.
        generation_idx : int
            index of what generation were currently on for multi token generation

    Attributes
    ----------
        value : str
            value store of Intervention
        id : str
            unique id of Intervention
        listeners : Set
            ids of Interventions that wish to be notified of a value change
        dependencies : Set
            ids of Interventions that this Intervention depends on having a set value
    '''

    interventions: Dict[str, Intervention] = OrderedDict()
    generation_idx: int = 0

    @classmethod
    def increment(cls) -> None:
        '''Increments Intervention.generation_idx by one.'''
        Intervention.generation_idx += 1



    @classmethod
    def parse(cls, arg: Value, promises: Dict[str, Dict]) -> Union[Intervention, Value]:
        '''
        Parses a promise and it's args into Interventions

        Parameters
        ----------
            arg : Value
                Value as argument of Intervention or id of arg Intervention to be converted into an Intervention
            promises : Dict
                Mapping of Promise id to Dict of attributes that make up a promise: 'args', 'command', and 'id'. Returned from Promise.compile()
        Returns
        ----------
            Union[Intervention,Value]
                arg Value or converted Intervention

        '''
        if isinstance(arg, str) and arg in promises:

            promise = promises[arg]
            promise['args'] = [Intervention.parse(
                arg, promises) for arg in promise['args']]
            promise = Intervention.create(**promise)

            return promise
        return arg

    @classmethod
    def from_execution_graph(cls, execution_graph: List[str], promises: Dict[str, Dict]) -> None:
        '''
        Parses the information from Promises into Interventions.
        Chains dependant Interventions.

        Parameters
        ----------
            execution_graph : List[str]
                List of ids of Promises to be executed in order of execution. Returned from Promise.compile()
            promises : Dict
                Mapping of Promise id to Dict of attributes that make up a promise: 'args', 'command', and 'id'. Returned from Promise.compile()
        '''

        dependancy = None

        for id in execution_graph:

            intervention = Intervention.parse(id, promises)
            if dependancy is not None:
                Chain(intervention, dependancy)
            dependancy = intervention

    @classmethod
    def create(cls, args: List[Union[Intervention, Value]], id: str, command: str) -> Intervention:
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

        return INTERVENTIONS_TYPES[command](*args, id)
    
    @classmethod
    def clear(cls) -> None:
        '''
        Clears the Intervention class attributes and clears subtypes.
        Sets Intervention.generation_idx to zero.
        '''
        Intervention.interventions.clear()
        Intervention.generation_idx = 0

        for type in Intervention.__subclasses__():

            type._clear()
    
    @abstractclassmethod
    def _clear(cls) -> None:
        '''Abstract method for subtypes to set when clearing information'''
        pass

    def __init__(self, id: str = None) -> None:

        self.value = None
        self.id = id or str(uuid.uuid4())
        self.listeners = set()
        self.dependencies = set()

        Intervention.interventions[self.id] = self

        self.listen(self)

    def cpu(self):

        def to_cpu(tensor: torch.Tensor):

            return tensor.cpu()

        self.value = apply(self.value, to_cpu, torch.Tensor)

        return self

    @abstractmethod
    def __call__(self):
        '''Abstract method for subtypes to set when performing their intervention'''
        pass

    def notify_listeners(self) -> None:
        '''
        Attempts to signal listener that it's value is changed.
        If the dependencies of a listener are fufilled, call the Intervention.
        Removes dependencies of listeners.
        '''

        for listener_id in list(self.listeners):

            if listener_id != self.id and listener_id in self.listeners:

                intervention = Intervention.interventions[listener_id]
                intervention.remove_dependency(self.id)

                if intervention.fufilled():
                    intervention()

    def fufilled(self) -> bool:
        '''
        Returns whether all dependencies have been fufilled, thefore if the 
        Intervention is ready to be executed.
        '''

        return len(self.dependencies) == 0

    def remove_dependency(self, id: str) -> None:
        '''
        Removes a dependency if it exists

        Parameters
        ----------
            id : str
                id of Intervention to remove from dependencies
        '''
        if id in self.dependencies:

            self.dependencies.remove(id)

    def depend(self, dependency: Intervention) -> None:
        '''
        Adds Intervention to dependencies and adds this Intervention to
        it's listeners.

        Parameters
        ----------
            dependency : Intervention
                dependency to add to dependencies
        '''

        self.dependencies.add(dependency.id)
        dependency.listen(self)

    def listen(self, listener: Intervention) -> None:
        '''
        Adds listener to listeners

        Parameters
        ----------
            listener : Intervention
                listener to add to listeners
        '''

        self.listeners.add(listener.id)

    def destroy(self) -> None:
        '''
        Removes reference to self from id to Intervention mapping and sets its self.value 
        to None.
        '''

        # Add back for debugging
        # print(f"Destroying {str(self)}")

        self.value = None

    def stop_listening(self, id: str) -> None:
        '''
        Removes Intervention with id from listeners. If there exist no more listeners,
        destory self.

        Parameters
        ----------
            id : str
                id of Intervention to remove from listeners
        '''

        self.listeners.remove(id)

        if len(self.listeners) == 0:

            self.destroy()

    def get_value(self, listener_id: str) -> torch.Tensor:
        '''
        Gets the Intervention's value. Requires an Intervention id in listeners.
        Removes the listener with id listener_id from listeners.
        If listener_id is not in listeners, raise ValueError
        If value is None, raise ValueError.

        Parameters
        ----------
            listener_id : str
                id of Intervention that requests value

        Returns
        ----------
            torch.Tensor
                value of Intervention
        '''

        if listener_id not in self.listeners:

            raise ValueError(
                f"Listener '{str(Intervention.interventions[listener_id])}' tried to reference value '{str(self)}' but not in listeners")

        if self.value is None:

            raise ValueError(
                f"Listener '{str(Intervention.interventions[listener_id])}' referenced value '{str(self)}' before assignment")

        value = self.value

        self.stop_listening(listener_id)

        return value

    def set_value(self, value: torch.Tensor, listener_id: str) -> None:
        '''
        Sets the Intervention's value. Requires an Intervention id in listeners.
        Removes the listener with id listener_id from listeners.
        Notifies listeners.

        Parameters
        ----------
            value : torch.Tensor
                value to set
            listener_id : str
                id of Intervention that requests to set value
        '''

        if listener_id is not None and listener_id not in self.listeners:

            raise ValueError(
                f"Listener '{str(Intervention.interventions[listener_id])}' tried to reference value '{str(self)}' but not in listeners")
        # Add back for debugging
        # print(f"Setting {self}")

        self.value = value

        self.stop_listening(listener_id)

        self.notify_listeners()


class Chain(Intervention):
    '''
    An Intervention to make one Intervention dependant on anothers fufillment.

    Attributes
    ----------
        arg1 : Intervention
            Intervention that is dependant on another Intervention
        arg2 : Intervention
            dependant Intervention

    '''

    def __init__(self, arg1: Intervention, arg2: Intervention, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.arg1 = arg1
        self.arg2 = arg2
        self.arg1.depend(self)
        self.depend(self.arg2)

    def __call__(self) -> None:

        self.arg2.stop_listening(self.id)

        self.notify_listeners()


class Add(Intervention):
    '''
    An Intervention to add two Interventions.

    Attributes
    ----------
        arg1 : Intervention
            first Intervention to add
        arg2 : Intervention
            second Intervention to add

    '''

    def __init__(self, arg1: Intervention, arg2: Intervention, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.arg1 = arg1
        self.arg2 = arg2
        self.depend(self.arg1)
        self.depend(self.arg2)

    def __repr__(self) -> str:
        return f"ADD({str(self.arg1)},{self.arg2})"

    def __call__(self) -> None:

        value = self.arg1.get_value(self.id) + self.arg2.get_value(self.id)

        self.set_value(value, self.id)


class Get(Intervention):
    '''
    An Intervention to get and store activations of a Module.

    Class Attributes
    ----------
        modules : Dict[str,Dict[str,Get]]
            mapping of module_name to: mapping of id to self. Used by intervention method to
            retrieve the correct Get Interventions that use a module

    Attributes
    ----------
        module_name : str
            module path requested for input or output
        batch_index : index
            index of Intervention within current batch
        generation_idx : index
            generation index that this Intervention should be executed

    '''
    modules: Dict[str, Dict[str, Get]] = dict()

    @classmethod
    def _clear(cls) -> None:
        Get.modules.clear()

    @classmethod
    def layers(cls) -> List[str]:

        return [module_name.replace('.input', '').replace('.output', '') for module_name in list(Get.modules.keys())]

    def __init__(self, module_name: str, batch_index: int, generation_idx: int, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.module_name = module_name
        self.batch_index = batch_index
        self.generation_idx = generation_idx

        if module_name in Get.modules:

            Get.modules[module_name][self.id] = self

        else:

            Get.modules[module_name] = {self.id: self}

    def __repr__(self) -> str:
        return f"GET({self.module_name}[{self.batch_index}])"

    def __call__(self, value: Tensor) -> None:

        # Add back for debugging
        # print(f'Reached {self.module_name}[{self.batch_index}]')

        self.value = self.batch_index_get(value)

        self.notify_listeners()

        self.batch_index_set(value, self.get_value(self.id))

        del Get.modules[self.module_name][self.id]

    def batch_index_set(self, value1, value2) -> None:
        if isinstance(value1, torch.Tensor):
            value1[[self.batch_index]] = value2
        elif isinstance(value1, list) or isinstance(value1, tuple):
            for value_idx in range(len(value1)):
                self.batch_index_set(value1[value_idx], value2[value_idx])

    def batch_index_get(self, value):
        if isinstance(value, torch.Tensor):
            return value[[self.batch_index]]
        elif isinstance(value, list) or isinstance(value, tuple):
            return [self.batch_index_get(_value) for _value in value]


class Set(Intervention):

    def __init__(self, arg1: Intervention, arg2: Intervention, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.arg1 = arg1
        self.arg2 = arg2
        self.depend(self.arg1)
        self.depend(self.arg2)

    def __repr__(self) -> str:
        return f"SET({str(self.arg1)},{self.arg2})"

    def __call__(self) -> None:

        self.arg1.set_value(self.arg2.get_value(self.id), self.id)

        self.notify_listeners()


class Copy(Intervention):

    copies: Dict[str, Copy] = dict()

    @classmethod
    def _clear(cls) -> None:
        for copy in Copy.copies.values():
            copy.copies = None
        Copy.copies.clear()

    def __init__(self, arg1: Intervention, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.arg1 = arg1
        self.depend(self.arg1)

        Copy.copies[self.id] = self

    def __repr__(self) -> str:
        return f"COPY({str(self.arg1)})"

    def __call__(self) -> None:

        value = self.arg1.get_value(self.id)
        self.set_value(value, self.id)

    def destroy(self) -> None:
        pass


class Slice(Intervention):

    def __init__(self, arg1: Intervention, slice, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.arg1 = arg1
        self.depend(self.arg1)

        self.slice = slice

    def __repr__(self) -> str:
        return f"Slice({self.arg1},{self.slice})"

    def __call__(self):

        value = self.arg1.get_value(self.id)[self.slice]
        self.set_value(value, self.id)


class Tensor(Intervention):

    tensors: Dict[str, Tensor] = dict()

    @classmethod
    def _clear(cls) -> None:
        for tensor in Tensor.tensors.values():
            tensor.value = None
        Tensor.tensors.clear()

    @classmethod
    def to(cls, device) -> None:
        for tensor in Tensor.tensors.values():
            # need to actually move tensors to model dtype
            tensor.value = tensor.value.to(device)

    def __init__(self, value: torch.Tensor, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.value = value

        Tensor.tensors[self.id] = self

    def __repr__(self) -> str:
        return f"TENSOR({self.id})"

    def listen(self, listener: Intervention) -> None:
        super().listen(listener)
        if self is not listener:
            listener.dependencies.remove(self.id)


class Adhoc(Intervention):

    model: torch.nn.Module = None
    adhoc_mode: bool = False

    @classmethod
    def _clear(cls) -> None:
        Adhoc.model = None

    def __init__(self, module_path: str, arg1: Intervention, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.arg1 = arg1
        self.depend(arg1)

        self.module_keys = module_path.replace(
            '[', '.').replace(']', '').split('.')

    def __call__(self) -> None:

        value = self.get_module()(self.arg1.get_value(self.id))

        self.set_value(value, self.id)

    @override
    def __enter__(self) -> None:
        Adhoc.adhoc_mode = True

    @override
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        Adhoc.adhoc_mode = False

    def get_module(self) -> torch.nn.Module:

        module = Adhoc.model

        for key in self.module_keys:

            if isinstance(module, list):
                module = module[int(key)]

            else:
                module = getattr(module, key)

        return module


INTERVENTIONS_TYPES.update(
    {'GET': Get, 'SET': Set, 'CPY': Copy, 'ADD': Add, 'TNS': Tensor, 'SLC': Slice, 'ADH': Adhoc})


def intervene(activations, module_name):

    if not Adhoc.adhoc_mode and module_name in Get.modules:

        for get in list(Get.modules[module_name].values()):

            if get.generation_idx == Intervention.generation_idx:

                get(activations)

    return activations


def output_intervene(activations, module_name):

    module_name = f"{module_name}.output"

    return intervene(activations, module_name)


def input_intervene(activations, module_name):

    module_name = f"{module_name}.input"

    return intervene(activations, module_name)
