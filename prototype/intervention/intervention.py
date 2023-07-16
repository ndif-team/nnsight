from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List

INTERVENTIONS = {}

class Intervention:

    interventions:Dict[str, Intervention] = dict()

    @classmethod
    def parse(cls, arg, promises):


        if isinstance(arg, str) and arg in promises:

            promise = promises[arg]
            promise['args'] = [Intervention.parse(arg, promises) for arg in promise['args']]

            return promise
        return arg


    @classmethod
    def from_execution_graph(cls, execution_graph, promises):

        subscriber = None

        for id in execution_graph:

            promise = Intervention.parse(id, promises)

            subscriber = Intervention.create(**promise, subscriber=subscriber)


    def __init__(self, id, subscriber=None) -> None:

        self._value = None
        self._id = id
        self._subscribers = []
        self.subscribe(subscriber)
        self._n_subscribers = len(self._subscribers)
        self._subscribing = True

        Intervention.interventions[self._id] = self

    @classmethod
    def create(cls, args, id, command, subscriber=None) -> None:

        if id in Intervention.interventions:

            intervention = Intervention.interventions[id]
            intervention.subscribe(subscriber)

            return intervention
                
        return INTERVENTIONS[command](*args,id, subscriber=subscriber)
    
    @abstractmethod
    def __call__(self):
        
        for subscriber in self._subscribers:

            if subscriber is not None and subscriber._subscribing:

                subscriber()

    def subscribe(self, subscriber):

        if subscriber is not None:

            self._subscribers.append(subscriber)

    def destroy(self):

        print(f"Destroying {type(self)}")

        del Intervention.interventions[self._id]

        self._value = None

    def reference(self):

        self._n_subscribers -= 1

        if self._n_subscribers == 0:

            self.destroy()

    @property
    def value(self):

        value = self._value

        self.reference()

        return value
    
    @value.setter
    def value(self, value): 
        self._value = value 
    
class Add(Intervention):

    def __init__(self, arg1, arg2, *args, **kwargs) -> None:

        super().__init__(*args,**kwargs)

        self.arg1 = Intervention.create(**arg1, subscriber=self)
        self.arg2 = Intervention.create(**arg2, subscriber=self)

    def __call__(self):

        if self.arg1._value is not None and self.arg2._value is not None:

            print('Adding')

            self._subscribing = False

            self.value = self.arg1.value + self.arg2.value

            super().__call__()

class Get(Intervention):

    gets:Dict[str, Get] = dict()

    @classmethod
    def layers(cls):

        return [layer.replace('.input', '').replace('.output', '') for layer in list(Get.gets.keys())]
    
    def __init__(self, module_name:str, *args, **kwargs) -> None:

        super().__init__(*args,**kwargs)

        self.module_name = module_name

        Get.gets[module_name] = self

    @Intervention.value.setter
    def value(self, value): 

        print(f"GSetting {self.module_name}")
        self._value = value 


        self()

class Set(Intervention):

    def __init__(self, arg1:Intervention, arg2:Intervention, *args, **kwargs) -> None:

        super().__init__(*args,**kwargs)

        self.arg1 = Intervention.create(**arg1, subscriber=self)
        self.arg2 = Intervention.create(**arg2, subscriber=self)

    def __call__(self) -> Any:

        if self.arg1._value is not None:

            print(f"Setting {self.arg1.module_name}")

            self._subscribing = False
    
            self.arg1.value = self.arg2.value
            self.arg1.reference()

            super().__call__()

            self.destroy()
        
class Copy(Intervention):

    copies:List[Copy] = list()

    def __init__(self, get:Dict, *args, **kwargs) -> None:

        super().__init__(*args,**kwargs)

        self.get = Intervention.create(**get, subscriber=self)

        Copy.copies.append(self)

    def __call__(self):

        if self.get._value is not None:

            print(f"Copying {self.get.module_name}")

            self._value = self.get.value

            self._subscribing = False

            super().__call__()

    @property
    def value(self):
        return self._value
    

class Slice(Intervention):

    def __init__(self, arg1:Dict, slice, *args, **kwargs) -> None:

        super().__init__(*args,**kwargs)

        self.arg1 = Intervention.create(**arg1, subscriber=self)
        self.slice = slice

    def __call__(self):

        print(f"Slicing")

        self._value = self.get.value[self.slice]

        self._subscribing = False

        super().__call__()


INTERVENTIONS.update({'GET': Get, 'SET': Set, 'CPY': Copy, 'ADD': Add})

def intervene(activations, module_name):

    if module_name in Get.gets:

        get = Get.gets[module_name]

        get.value = activations

        return get._value
    
    return activations


def output_intervene(activations, module_name):

    module_name = f"{module_name}.output"

    return intervene(activations, module_name)


def input_intervene(activations, module_name):

    module_name = f"{module_name}.input"

    return intervene(activations,module_name)
