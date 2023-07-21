from __future__ import annotations

import uuid
from abc import abstractclassmethod, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, MutableMapping, Union

import torch
from typing_extensions import override

from ..util import Value

INTERVENTIONS_TYPES = {}


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
        listeners : Set
            ids of Interventions that wish to be notified of a value change
        current_listeners : Set
            ids of listeners that have yet to be notified
        dependencies : Set
            ids of Interventions that this Intervention depends on having a value
        current_dependencies : Set
            ids of dependencies that have yet to be fufilled
    '''

    interventions: Dict[str, Intervention] = OrderedDict()

    @classmethod
    def clear(cls) -> None:
        '''Clears the Intervention class attributes and clears subtypes'''
        Intervention.interventions.clear()

        for type in Intervention.__subclasses__():

            type._clear()

    @abstractclassmethod
    def _clear(cls) -> None:
        '''Abstract method for subtypes to set when clearing information'''
        pass

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
        Chains dependant Interventions

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
    def reset(cls) -> None:
        '''Resets class attributes and resets subtypes'''
        for intervention in Intervention.interventions.values():
            intervention.current_listeners.update(intervention.listeners)
            intervention.current_dependencies.update(intervention.dependencies)

        for type in Intervention.__subclasses__():

            type._reset()

    @abstractclassmethod
    def _reset(cls) -> None:
        '''Abstract method for subtypes to set when resetting information'''
        pass

    def __init__(self, id: str = None) -> None:

        self._value = None
        self.id = id or str(uuid.uuid4())
        self.listeners = set()
        self.current_listeners = set()
        self.dependencies = set()
        self.current_dependencies = set()

        Intervention.interventions[self.id] = self

        self.listen(self)

    @abstractmethod
    def __call__(self):
        '''Abstract method for subtypes to set when performing their intervention'''
        pass

    def notify_listeners(self) -> None:
        '''
        Abstract method that attempts to signal listener that it's value is changed.
        If the dependencies of a listener are fufilled, call the Intervention.
        Removes dependencies of listeners.
        '''

        for listener_id in list(self.current_listeners):

            if listener_id != self.id and listener_id in self.current_listeners:

                intervention = Intervention.interventions[listener_id]
                intervention.remove_dependency(self.id)

                if intervention.fufilled():
                    intervention()

    def fufilled(self) -> bool:
        '''
        Returns wheather all dependencies have been fufilled, thefore the 
        Intervention is ready to be executed.
        '''

        return len(self.current_dependencies) == 0

    def remove_dependency(self, id: str):
        '''
        Removes a dependency if it exists

        Parameters
        ----------
            id : str
                id of Intervention to remove from dependencies
        '''
        if id in self.current_dependencies:

            self.current_dependencies.remove(id)

    def depend(self, dependency: Intervention):
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
        Removes reference to self from id to Intervention mapping and sets its self._value 
        to None.
        '''

        print(f"Destroying {str(self)}")

        self._value = None

    def stop_listening(self, id: str) -> None:
        '''
        Removes Intervention with id from listeners. If there exist no more listeners,
        destory self.

        Parameters
        ----------
            id : str
                id of Intervention to remove from listeners
        '''

        self.current_listeners.remove(id)

        if len(self.current_listeners) == 0:

            self.destroy()

    def get_value(self, listener_id: str) -> torch.Tensor:
        '''
        Gets the Intervention's value. Requires an Intervention id in listeners.
        Removes the listener with id listener_id from listeners.
        If listener_id is not in current_listeners, raise ValueError
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

        if listener_id not in self.current_listeners:

            raise ValueError(
                f"Listener '{str(Intervention.interventions[listener_id])}' tried to reference value '{str(self)}' but not in listeners")

        if self._value is None:

            raise ValueError(
                f"Listener '{str(Intervention.interventions[listener_id])}' referenced value '{str(self)}' before assignment")

        value = self._value

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

        if listener_id is not None and listener_id not in self.current_listeners:

            raise ValueError(
                f"Listener '{str(Intervention.interventions[listener_id])}' tried to reference value '{str(self)}' but not in listeners")

        print(f"Setting {self}")

        self._value = value

        self.stop_listening(listener_id)

        self.notify_listeners()


class Chain(Intervention):
    '''
    An Intervention to make one Intervention depandant on anothers fufillment.

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

    def __call__(self):

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

    def __call__(self):

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
        current_modules : Dict[str,Dict[str,Get]]
            module mappings that haven't been removed for a run

    Attributes
    ----------
        module_name : str
            module path requested for input or output
        batch_index : index
            index of Intervention within current batch

    '''
    modules: Dict[str, Dict[str, Get]] = dict()
    current_modules: Dict[str, Dict[str, Get]] = dict()

    @classmethod
    def _clear(cls) -> None:
        Get.modules.clear()
        Get.current_modules.clear()

    @classmethod
    def _reset(cls) -> None:
        Get.current_modules.update(Get.modules)

    @classmethod
    def layers(cls) -> List[str]:

        return [module_name.replace('.input', '').replace('.output', '') for module_name in list(Get.modules.keys())]

    def __init__(self, module_name: str, batch_index: int, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.module_name = module_name
        self.batch_index = batch_index

        if module_name in Get.modules:

            Get.modules[module_name][self.id] = self

        else:

            Get.modules[module_name] = {self.id: self}

    def __repr__(self) -> str:
        return f"GET({self.module_name}[{self.batch_index}])"

    def __call__(self, value: Tensor) -> None:

        print(f'Reached {self.module_name}[{self.batch_index}]')

        self._value = self.batch_index_get(value)

        self.notify_listeners()

        self.batch_index_set(value, self.get_value(self.id))

        del Get.current_modules[self.module_name][self.id]


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

    def __call__(self) -> Any:

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

        self.copies = []

        Copy.copies[self.id] = self

    def __repr__(self) -> str:
        return f"COPY({str(self.arg1)})"

    def __call__(self):

        value = self.arg1.get_value(self.id)
        self.set_value(value, self.id)

        self.copies.append(value)


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
            tensor._value = None
        Tensor.tensors.clear()

    @classmethod
    def to(cls, device):
        for tensor in Tensor.tensors.values():
            # need to actually move tensors to model dtype
            tensor._value = tensor._value.to(device).half()

    def __init__(self, value: torch.Tensor, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self._value = value

        Tensor.tensors[self.id] = self

    def __repr__(self) -> str:
        return f"TENSOR({self.id})"

    def listen(self, listener: Intervention) -> None:
        super().listen(listener)
        if self is not listener:
            listener.dependencies.remove(self.id)

class Adhoc(Intervention):

    model:torch.nn.Module = None
    adhoc_mode:bool = False

    @classmethod
    def _clear(cls) -> None:
        Adhoc.model = None

    def __init__(self, module_path:str, arg1:Intervention, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.arg1 = arg1
        self.depend(arg1)

        self.module_keys = module_path.replace('[', '.').replace(']','').split('.')

    def __call__(self):
        
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
    {'GET': Get, 'SET': Set, 'CPY': Copy, 'ADD': Add, 'TNS': Tensor, 'SLC': Slice, 'ADH' : Adhoc})


def intervene(activations, module_name):

    if not Adhoc.adhoc_mode and module_name in Get.current_modules:

        for get in list(Get.current_modules[module_name].values()):

            get(activations)

    return activations


def output_intervene(activations, module_name):

    module_name = f"{module_name}.output"

    return intervene(activations, module_name)


def input_intervene(activations, module_name):

    module_name = f"{module_name}.input"

    return intervene(activations, module_name)
