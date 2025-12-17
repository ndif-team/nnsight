from typing import Any, Tuple, List, Optional, Union

import torch
from ..util import apply, applyn


class Batchable:

    ### Abstract methods ###

    def _prepare_input(self, *inputs, **kwargs):

        return inputs, kwargs

    def _batch(self, batched_input, *args, **kwargs):

        raise NotImplementedError(
            "Batching not implemented for this model and is required for multiple invokers"
        )


class Batcher:

    def __init__(self, *args, **kwargs):

        self.batched_args = args
        self.batched_kwargs = kwargs

        self.last_batch_group: Optional[List[int]] = None
        self.needs_batching = False

        self.current_value: Optional[Any] = None

    @property
    def total_batch_size(self):

        return sum(self.last_batch_group)

    def batch(
        self, batchable: Batchable, *args, **kwargs
    ) -> Tuple[Tuple[Any, Any], Optional[List[int]]]:

        if args or kwargs:

            args, kwargs = batchable._prepare_input(*args, **kwargs)

            if self.last_batch_group is None:
                self.batched_args = args
                self.batched_kwargs.update(kwargs)

                self.last_batch_group = [-1, -1]

            else:

                if self.last_batch_group == [-1, -1]:
                    (self.batched_args, self.batched_kwargs), batch_size = (
                        batchable._batch(
                            None, *self.batched_args, **self.batched_kwargs
                        )
                    )

                    self.last_batch_group[0] = 0
                    self.last_batch_group[1] = batch_size

                (self.batched_args, self.batched_kwargs), batch_size = batchable._batch(
                    (self.batched_args, self.batched_kwargs), *args, **kwargs
                )

                self.last_batch_group = [sum(self.last_batch_group), batch_size]

                self.needs_batching = True

            return (args, kwargs), self.last_batch_group

        return (args, kwargs), None

    def narrow(self, batch_group: Optional[List[int]]):

        data = self.current_value

        if not self.needs_batching or batch_group is None:
            return data

        batch_start, batch_size = batch_group

        if batch_start == -1:
            return data

        def _narrow(acts: torch.Tensor):

            if acts.shape[0] == self.total_batch_size:
                return acts.narrow(0, batch_start, batch_size)

            return acts

        return apply(
            data,
            _narrow,
            torch.Tensor,
        )

    def swap(self, batch_group: Optional[List[int]], swap_value: Any):

        if not self.needs_batching or batch_group is None:
            self.current_value = swap_value
            return

        batch_start, batch_size = batch_group

        if batch_start == -1:
            self.current_value = swap_value
            return

        def _swap(current_value: torch.Tensor, swap_value: torch.Tensor):

            if current_value.shape[0] == self.total_batch_size:

                needs_concat = (
                    current_value.requires_grad and current_value.is_leaf
                ) or current_value._base is not None

                if needs_concat:
                    pre = current_value.narrow(0, 0, batch_start)
                    post = (
                        current_value.narrow(
                            0,
                            batch_start + batch_size,
                            current_value.shape[0] - batch_start - batch_size,
                        )
                        if self.total_batch_size == current_value.shape[0]
                        else current_value
                    )

                    return torch.cat([pre, swap_value, post], dim=0)

                else:
                    current_value[batch_start : batch_start + batch_size] = swap_value

            return current_value

        self.current_value = applyn(
            [self.current_value, swap_value], _swap, torch.Tensor
        )
