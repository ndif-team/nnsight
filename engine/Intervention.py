from __future__ import annotations

from typing import Any, Callable, Dict, List, Set, Tuple, Union

import torch
import torch.futures
import torch.fx
from torch.utils.hooks import RemovableHandle

from . import logger, util
from .fx.Proxy import InterventionProxy
from .modeling import InterventionModel


class InterventionTree:
    """
    Attributes:
        interventions (Dict[str, Intervention]): _description_
        module_path_to_intervention_name (Dict[str, str): _description_
        modules (Set[str]): _description_
        generation_idx (int): _description_
        model: _description_
    """

    @staticmethod
    def from_pydantic(pinterventions: Dict[str, InterventionModel]) -> InterventionTree:
        """
        Creates an InterventionTree and its associated Interventions from a dictionary of
        InterventionModel pydantic models.

        Args:
            pinterventions (Dict[str, InterventionModel]): _description_

        Returns:
            InterventionTree: _description_
        """

        tree = InterventionTree()

        for node in pinterventions.values():
            tree._from_pydantic(node, pinterventions)
        return tree

    def _from_pydantic(
        self,
        pintervention: InterventionModel,
        pinterventions: Dict[str, InterventionModel],
    ) -> Intervention:
        """Creates an Intervention from a pydantic model

        Args:
            pintervention (InterventionModel): _description_
            pinterventions (Dict[str, InterventionModel]): _description_

        Returns:
            Intervention: _description_
        """

        def _dereference(reference: InterventionModel.Reference):
            return self._from_pydantic(pinterventions[reference.name], pinterventions)

        # Arguments might be interventions themselves so recurse.
        args = util.apply(pintervention.args, _dereference, InterventionModel.Reference)
        kwargs = util.apply(
            pintervention.kwargs, _dereference, InterventionModel.Reference
        )

        # Processing of args may have already created an Intervention for this node so just return it.
        if pintervention.name in self.interventions:
            return self.interventions[pintervention.name]

        intervention = self.create(
            pintervention.operation,
            pintervention.name,
            pintervention.target,
            *args,
            **kwargs,
        )

        self.interventions[intervention.name] = intervention

        # Dependencies and listeners might be interventions themselves so recurse.
        dependencies = util.apply(
            pintervention.dependencies, _dereference, InterventionModel.Reference
        )
        listeners = util.apply(
            pintervention.listeners, _dereference, InterventionModel.Reference
        )

        # Set the callbacks for dependencies and listeners.
        intervention.set_listeners(listeners)
        intervention.set_dependencies(dependencies)

        return intervention

    def __init__(self) -> None:
        self.interventions: Dict[str, Intervention] = dict()
        self.module_path_to_intervention_name: Dict[str, str] = dict()
        self.modules: Set[str] = set()
        self.generation_idx: int = 0
        self.model: torch.nn.Module = None

    def set_model(self, model: torch.nn.Module) -> RemovableHandle:
        """
        Adds a reference to a model to the tree so ModuleInterventions can access
        its modules and run them. Also hooks the model so after each time it is ran, we can keep track
        of the generation_idx

        Args:
            model (torch.nn.Module): _description_

        Returns:
            RemovableHandle: Returns the hook so it can be removed after completion.
        """
        self.model = model

        return self.model.register_forward_hook(
            lambda module, input, output: self.increment()
        )

    def increment(self) -> None:
        """Increments generation_idx by one."""
        self.generation_idx += 1

    def create(
        self,
        operation: str,
        name: str,
        target: Union[Callable[..., Any], str],
        *args,
        **kwargs,
    ) -> Intervention:
        """Creates the appropratie type of intervention based on its attributes.

        Args:
            operation (str): _description_
            name (str): _description_
            target (Union[Callable[..., Any], str]): _description_

        Returns:
            Intervention: _description_
        """
        if name in self.interventions:
            return self.interventions[name]
        if operation == "placeholder":
            intervention = ActivationIntervention(
                self, operation, name, target, *args, **kwargs
            )
            self.modules.add(".".join(intervention.target.split(".")[:-3]))
            self.module_path_to_intervention_name[
                intervention.target
            ] = intervention.name
            return intervention
        if operation == "call_module":
            return ModuleIntervention(self, operation, name, target, *args, **kwargs)
        if target is InterventionProxy.proxy_save:
            return SaveIntervention(self, operation, name, target, *args, **kwargs)
        if target is InterventionProxy.proxy_set:
            return SetIntervention(self, operation, name, target, *args, **kwargs)

        return Intervention(self, operation, name, target, *args, **kwargs)


class Intervention(torch.futures.Future):
    """
    An Intervention represents some action that needs to be carried out during the inference of a model.

    Attributes:
        tree (InterventionTree): _description_
        operation (str): _description_
        name (str): _description_
        target(Union[Callable[..., Any], str]): _description_
        args (List[Any]): _description_
        kwargs (Dict[str,Any]): _description_
    """

    def __init__(
        self,
        tree: InterventionTree,
        operation: str,
        name: str,
        target: Union[Callable[..., Any], str],
        *args,
        **kwargs,
    ):
        super().__init__()

        self.tree = tree

        self.operation = operation
        self.name = name
        self.target = target
        self.args = args
        self.kwargs = kwargs

        self.fufilled = None

        # Add done callback to self so we see in logs when this is set.
        self.add_done_callback(lambda x: logger.debug(f"=> SET({self.name})"))

    def set_listeners(self, listeners: List[Intervention]) -> None:
        """
        Chains listeners to have their .chain() method called in order of listeners.
        Collects all listeners as a single Future and adds a done callback to .destroy()
        when completed.

        Args:
            listeners (List[Intervention]): _description_
        """

        future = self

        for listener in listeners:
            future = future.then(listener.chain)

        # Add self to listeners becuase we should only destory this Intervention after it's been set.
        listeners.append(self)

        torch.futures.collect_all(listeners).add_done_callback(lambda x: self.destroy())

    def set_dependencies(self, dependencies: List[Intervention]) -> None:
        """
        Sets the .fufilled attribute to be a single future collected from all dependecies.
        .fufilled.done() tells you whether the Intervention is ready to be ran.

        Args:
            dependencies (List[Intervention]): _description_
        """

        self.fufilled = torch.futures.collect_all(dependencies)

    def prepare_inputs(self) -> Tuple[List[Any], Dict[str, Any]]:
        """Preprocess this interventions input to be ran by its command

        Returns:
            Tuple[List[Any], Dict[str,Any]]: _description_
        """

        # Turn futures into their value
        def _value(value: torch.futures.Future):
            return value.value()

        # Move tensors to device
        def _to(value: torch.Tensor):
            return value.to(self.tree.model.device)

        args = util.apply(self.args, _value, torch.futures.Future)
        args = util.apply(args, _to, torch.Tensor)

        kwargs = util.apply(self.kwargs, _value, torch.futures.Future)
        kwargs = util.apply(kwargs, _to, torch.Tensor)

        return args, kwargs

    def intervene(self) -> None:
        """Carries out the Intervention and sets the result."""
        args, kwargs = self.prepare_inputs()

        output = self.target(*args, **kwargs)

        self.set_result(output)

    def chain(self, future: Intervention):
        if self.fufilled.done():
            self.intervene()

        future.set_result(None)

    def destroy(self) -> None:
        """Destroys the Intervention"""
        logger.debug(f"=> DEL({self.name})")

        del self.tree.interventions[self.name]

    @staticmethod
    def update(value1, value2) -> None:
        if isinstance(value1, torch.Tensor):
            value1[:] = value2
        elif isinstance(value1, list) or isinstance(value1, tuple):
            for value_idx in range(len(value1)):
                Intervention.update(value1[value_idx], value2[value_idx])
        elif isinstance(value1, dict):
            for key in value1:
                Intervention.update(value1[key], value2[key])


class ActivationIntervention(Intervention):
    def intervene(self):
        pass

    def batch_index_set(self, batch_index: int, value) -> None:
        def _batch_index_set(value):
            return value[[batch_index]]

        self.set_result(util.apply(value, _batch_index_set, torch.Tensor))


class SetIntervention(Intervention):
    def intervene(self):
        (value1, value2), _ = self.prepare_inputs()

        Intervention.update(value1, value2)

        self.destroy()


class SaveIntervention(Intervention):
    def intervene(self):
        (value, *_), _ = self.prepare_inputs()

        self.set_result(util.apply(value, lambda x: x.clone(), torch.Tensor))

    def destroy(self):
        pass


class ModuleIntervention(Intervention):
    def intervene(self):
        args, kwargs = self.prepare_inputs()

        module = util.fetch_attr(self.tree.model, self.target)

        output = module(*args, **kwargs)

        self.set_result(output)


def intervene(activations, module_path: str, tree: InterventionTree, key: str):
    batch_idx = 0

    module_path = f"{module_path}.{key}.{tree.generation_idx}"

    batch_module_path = f"{module_path}.{batch_idx}"

    while batch_module_path in tree.module_path_to_intervention_name:
        intervention: ActivationIntervention = tree.interventions[
            tree.module_path_to_intervention_name[batch_module_path]
        ]

        intervention.batch_index_set(batch_idx, activations)

        batch_idx += 1

        batch_module_path = f"{module_path}.{batch_idx}"

    return activations
