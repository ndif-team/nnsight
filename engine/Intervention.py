from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import torch
import torch.futures
import torch.fx
from torch.utils.hooks import RemovableHandle

from . import logger, util
from .fx import Proxy
from .modeling import InterventionModel


class InterventionTree:
    """

    Attributes:
        interventions (Dict[str, Intervention]): _description_
        activations (Dict[str, ActivationIntervention]): _description_
        modules (Set[str]): _description_
        generation_idx (int): _description_
    """

    @staticmethod
    def from_pydantic(interventions: Dict[str, InterventionModel]) -> InterventionTree:
        """_summary_

        Args:
            graph (torch.fx.graph.Graph): _description_
        """

        tree = InterventionTree()

        for node in interventions.values():
            tree._from_pydantic(node, interventions)

        return tree

    def _from_pydantic(
        self,
        pintervention: InterventionModel,
        interventions: Dict[str, InterventionModel],
    ) -> Intervention:
        """Creates an Intervention from a Node.

        Args:
            node (torch.fx.node.Node): _description_

        Returns:
            Intervention: _description_
        """

        def _dereference(reference: InterventionModel.Reference):
            return self._from_pydantic(interventions[reference.name], interventions)

        # Arguments might be nodes themselves so recurse.
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

        # Dependencies and listeners might be nodes themselves so recurse.
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
        self.activation_interventions: Dict[str, ActivationIntervention] = dict()
        self.save_interventions: Dict[str, ActivationIntervention] = dict()
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
        """_summary_

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
            return ActivationIntervention(
                self, operation, name, target, *args, **kwargs
            )
        if operation == "call_module":
            return ModuleIntervention(self, operation, name, target, *args, **kwargs)
        if target is Proxy.proxy_save:
            return SaveIntervention(self, operation, name, target, *args, **kwargs)
        if target is Proxy.proxy_set:
            return SetIntervention(self, operation, name, target, *args, **kwargs)

        return Intervention(self, operation, name, target, *args, **kwargs)


class Intervention(torch.futures.Future):
    """
    An Intervention represents some action that needs to be carried out during the inference of a model.

    Attributes:
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
        """
        Args:
            operation (str): _description_
            name (str): _description_
            target (Union[Callable[..., Any], str]): _description_
        """
        super().__init__()

        self.tree = tree

        self.operation = operation
        self.name = name
        self.target = target
        self.args = args
        self.kwargs = kwargs

        # Add to class attribute Intervention.interventions
        tree.interventions[name] = self

        # Add done callback to self so we see in logs when this is set.
        self.add_done_callback(lambda x: logger.debug(f"=> SET({self.name})"))

    def set_listeners(self, listeners: List[Intervention]) -> None:
        """
        Collects all listeners as a single Future and adds a done call back to call destroy()
        when completed.

        Args:
            listeners (List[Intervention]): _description_
        """

        # Add self to listeners becuase we should only destory this Intervention after it's been set.
        listeners.append(self)
        torch.futures.collect_all(listeners).add_done_callback(lambda x: self.destroy())

    def set_dependencies(self, dependencies: List[Intervention]) -> None:
        """
        Collects all dependencies (arguments that are also Interventions) as a single Future and adds a done call back to call intervene()
        when completed.

        Args:
            dependencies (List[Intervention]): _description_
        """
        torch.futures.collect_all(dependencies).add_done_callback(
            lambda x: self.intervene()
        )

    def prepare_inputs(self) -> Tuple[List[Any], Dict[str, Any]]:
        """Preprocess this interventions input to be ran by its command

        Returns:
            Tuple[List[Any], Dict[str,Any]]: _description_
        """

        def _intervene(value: torch.futures.Future):
            return value.value()

        # TODO make this dynamic
        device = self.tree.model.device

        def _to(value: torch.Tensor):
            return value.to(device)

        # Turn futures into their values
        args = util.apply(self.args, _intervene, torch.futures.Future)
        kwargs = util.apply(self.kwargs, _intervene, torch.futures.Future)
        # Move tensors to meta device
        args = util.apply(args, _to, torch.Tensor)
        kwargs = util.apply(kwargs, _to, torch.Tensor)

        return args, kwargs

    def intervene(self) -> None:
        """Carries out the Intervention and sets the result."""
        args, kwargs = self.prepare_inputs()

        output = self.target(*args, **kwargs)

        self.set_result(output)

    def destroy(self) -> None:
        """Destroys the Intervention"""
        logger.debug(f"=> DEL({self.name})")

        del self.tree.interventions[self.name]


class ActivationIntervention(Intervention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tree.modules.add(".".join(self.target.split(".")[:-3]))
        self.tree.activation_interventions[self.target] = self

    def set_dependencies(self, dependencies: List[Intervention]):
        return super().set_dependencies([self])

    def intervene(self):
        pass

    @staticmethod
    def batch_index_update(batch_index: int, value1, value2) -> None:
        if isinstance(value1, torch.Tensor):
            value1[[batch_index]] = value2
        elif isinstance(value1, list) or isinstance(value1, tuple):
            for value_idx in range(len(value1)):
                ActivationIntervention.batch_index_update(batch_index, value1[value_idx], value2[value_idx])
        elif isinstance(value1, dict):
            for key in value1:
                ActivationIntervention.batch_index_update(batch_index, value1[key], value2[key])

    def batch_index_set(self, batch_index: int, value) -> None:
        def _batch_index_set(value):
            return value[[batch_index]]

        self.set_result(util.apply(value, _batch_index_set, torch.Tensor))


class SetIntervention(Intervention):
    def intervene(self):
        module_name, activation_intervention, value_intervention = self.args
        self.tree.activation_interventions[module_name] = value_intervention

        self.set_result(value_intervention.value())


class SaveIntervention(Intervention):
    def intervene(self):
        intervention = self.args[0]

        self.tree.save_interventions[self.name] = self

        self.set_result(copy.deepcopy(intervention.value()))

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

    while batch_module_path in tree.activation_interventions:
        intervention = tree.activation_interventions[batch_module_path]

        intervention.batch_index_set(batch_idx, activations)

        _intervention = tree.activation_interventions[batch_module_path]

        if intervention is not _intervention:

            ActivationIntervention.batch_index_update(
                batch_idx, activations, _intervention.value()
            )

        batch_idx += 1

        batch_module_path = f"{module_path}.{batch_idx}"

    return activations
