from functools import wraps
import os
from typing import Dict, Union

import yaml
import torch
from .patching import *
from .pydantics.Config import ConfigModel

PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, "config.yaml"), "r") as file:
    CONFIG = ConfigModel(**yaml.safe_load(file))

from .logger import logger
from .models.NNsightModel import NNsight
from .models.LanguageModel import LanguageModel

from .patching import Patch, Patcher
from .tracing.Proxy import proxy_wrapper

logger.disabled = not CONFIG.APP.LOGGING

# Below do default patching:
DEFAULT_PATCHER = Patcher()

from inspect import getmembers, isfunction

import einops

for key, value in getmembers(einops.einops, isfunction):
    DEFAULT_PATCHER.add(Patch(einops.einops, proxy_wrapper(value), key))

# TODO THis does not work. Because of accelerate also patching? because they are overloaded?
#DEFAULT_PATCHER.add(Patch(torch, proxy_wrapper(torch.zeros), "zeros"))
# DEFAULT_PATCHER.add(Patch(torch, proxy_wrapper(torch.ones), "ones"))
# DEFAULT_PATCHER.add(Patch(torch, proxy_wrapper(torch.rand), "rand"))

from torch._subclasses.fake_tensor import FakeTensor


def _bool(self):
    return True


DEFAULT_PATCHER.add(Patch(FakeTensor, _bool, "__bool__"))


def fake_tensor_new_wrapper(fn):

    @wraps(fn)
    def inner(cls, fake_mode, elem, device, constant=None):

        if isinstance(elem, FakeTensor):

            return elem

        else:

            return fn(cls, fake_mode, elem, device, constant=constant)

    return inner


DEFAULT_PATCHER.add(
    Patch(FakeTensor, fake_tensor_new_wrapper(FakeTensor.__new__), "__new__")
)


def onehot_wrapper(fn):
    @wraps(fn)
    def onehot(input: torch.Tensor, num_classes=-1):
        if input.device.type == "meta":
            return torch.zeros((*input.shape, num_classes), device="meta")

        else:
            return fn(input, num_classes=num_classes)

    return onehot


DEFAULT_PATCHER.add(
    Patch(torch.nn.functional, onehot_wrapper(torch.nn.functional.one_hot), "one_hot")
)


def noop_wrapper(fn):
    @wraps(fn)
    def noop(input: torch.Tensor, *args, **kwargs):
        return input

    return noop


DEFAULT_PATCHER.add(Patch(FakeTensor, noop_wrapper(FakeTensor.tolist), "tolist"))

import warnings

try:
    # Hacky patch to get around the fact this init method has no handling for 'meta' tensors.
    def autoamp_init(
        self,
        device_type: str,
        dtype=None,
        enabled: bool = True,
        cache_enabled: Optional[bool] = None,
    ):
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = device_type
            self.fast_dtype = dtype
            # TODO: support get_autocast_gpu/cpu_dtype
            assert dtype is not None
            return
        self.device = device_type
        self.custom_backend_name = torch._C._get_privateuse1_backend_name()

        if self.device == "cuda":
            self.fast_dtype = torch.get_autocast_gpu_dtype()
        ### PATCH ###
        elif self.device == "meta":
            self.fast_dtype = torch.get_autocast_cpu_dtype()
        ### PATCH ###
        elif self.device == "cpu":
            self.fast_dtype = torch.get_autocast_cpu_dtype()
        elif self.device == "xpu":
            self.fast_dtype = torch.xpu.get_autocast_xpu_dtype()  # type: ignore[attr-defined]
        elif self.device == "ipu":
            self.fast_dtype = torch.get_autocast_ipu_dtype()  # type: ignore[attr-defined]
        elif self.device == "hpu":
            self.fast_dtype = torch.hpu.get_autocast_hpu_dtype()  # type: ignore[attr-defined]
        elif self.device == "xla":
            self.fast_dtype = torch.get_autocast_xla_dtype()  # type: ignore[attr-defined]
        elif self.device == self.custom_backend_name:
            necessary_funcs = [
                "is_autocast_enabled",
                "set_autocast_enabled",
                "get_autocast_dtype",
                "set_autocast_dtype",
                "get_amp_supported_dtype",
            ]
            message = f"Tried to use AMP with the `{self.custom_backend_name}` backend, but the backend has not "
            message += "registered a module or  the module miss some necessary funcs. The backend should register "
            message += "a module by `torch._register_device_module`, and the module must have these funcs: \n"
            message += "`is_autocast_enabled() -> bool`, `set_autocast_enabled(bool) -> None`, "
            message += "`get_autocast_dtype() -> torch.dtype`, `set_autocast_dtype(torch.dtype) "
            message += (
                "-> None` and `get_amp_supported_dtype() -> List[torch.dtype]`. \n"
            )

            assert hasattr(torch, self.custom_backend_name), message
            self.custom_device_mod = getattr(torch, self.custom_backend_name)
            for func in necessary_funcs:
                assert hasattr(self.custom_device_mod, func), (
                    message + f"But the func `{func}` is missing. \n"
                )

            self.fast_dtype = self.custom_device_mod.get_autocast_dtype()
        else:
            raise RuntimeError(
                f"User specified an unsupported autocast device_type '{self.device}'"
            )
        self._cache_enabled = torch.is_autocast_cache_enabled()
        if (
            enabled
            and torch.cuda.amp.common.amp_definitely_not_available()
            and self.device == "cuda"
        ):
            warnings.warn(
                "User provided device_type of 'cuda', but CUDA is not available. Disabling"
            )
            enabled = False
        if dtype is not None:
            self.fast_dtype = dtype
        if cache_enabled is not None:
            self._cache_enabled = cache_enabled

        if self.device == "cpu":
            supported_dtype = [torch.bfloat16, torch.float16]
            if self.fast_dtype not in supported_dtype and enabled:
                error_message = "In CPU autocast, but the target dtype is not supported. Disabling autocast.\n"
                error_message += "CPU Autocast only supports dtype of "
                error_message += (
                    ", ".join(str(dtype) for dtype in supported_dtype) + " currently."
                )
                warnings.warn(error_message)
                enabled = False
        elif self.device == "xpu":
            supported_dtype = [torch.bfloat16, torch.float16]
            if self.fast_dtype not in supported_dtype:
                error_message = "In XPU autocast, but the target dtype is not supported. Disabling autocast.\n"
                error_message += "XPU Autocast only supports dtypes of torch.bfloat16 and torch.float16 currently."
                warnings.warn(error_message)
                enabled = False
        elif self.device == "ipu":
            supported_dtypes = [torch.bfloat16, torch.float16]
            if self.fast_dtype not in supported_dtypes:
                error_message = "In IPU autocast, but the target dtype is not supported. Disabling autocast.\n"
                error_message += "IPU Autocast only supports dtypes of torch.bfloat16 and torch.float16 currently."
                warnings.warn(error_message)
                enabled = False
        elif self.device == "hpu":
            supported_dtype = [torch.bfloat16, torch.float16]
            if self.fast_dtype not in supported_dtype:
                error_message = "In HPU autocast, but the target dtype is not supported. Disabling autocast.\n"
                error_message += "HPU Autocast only supports dtypes of torch.bfloat16 and torch.float16 currently."
                warnings.warn(error_message)
                enabled = False
        elif self.device == self.custom_backend_name:
            supported_dtype = self.custom_device_mod.get_amp_supported_dtype()
            if self.fast_dtype not in supported_dtype:
                error_message = f"In {self.custom_backend_name} autocast, but the target dtype is not supported. "
                error_message += f"Disabling autocast.\n {self.custom_backend_name} Autocast only supports dtypes of "
                error_message += (
                    ", ".join(str(dtype) for dtype in supported_dtype) + " currently."
                )
                warnings.warn(error_message)
                enabled = False
        elif self.device == "cuda":
            if (
                enabled
                and self.fast_dtype == torch.bfloat16
                and not torch.cuda.is_bf16_supported()
            ):
                raise RuntimeError(
                    "Current CUDA Device does not support bfloat16. Please switch dtype to float16."
                )
        elif self.device == "xla":
            supported_dtype = [torch.float16, torch.bfloat16]
            if self.fast_dtype not in supported_dtype:
                error_message = "In XLA autocast, but the target dtype is not supported. Disabling autocast.\n"
                error_message += (
                    "XLA Autocast only supports dtype of torch.bfloat16 currently."
                )
                warnings.warn(error_message)
                enabled = False
        self._enabled = enabled

    from torch.amp.autocast_mode import autocast

    DEFAULT_PATCHER.add(Patch(autocast, autoamp_init, "__init__"))

except:
    pass

try:

    from accelerate.utils.modeling import (
        is_npu_available,
        check_device_same,
        is_xpu_available,
    )

    # Hacky patch to get around this function trying to set the parameter of a non meta tensor to meta.
    # Also handles FakeTensors.
    def set_module_tensor_to_device(
        module: torch.nn.Module,
        tensor_name: str,
        device: Union[int, str, torch.device],
        value: Optional[torch.Tensor] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        fp16_statistics: Optional[torch.HalfTensor] = None,
        tied_params_map: Optional[Dict[int, Dict[torch.device, torch.Tensor]]] = None,
    ):
        """
        A helper function to set a given tensor (parameter of buffer) of a module on a specific device (note that doing
        `param.to(device)` creates a new tensor not linked to the parameter, which is why we need this function).

        Args:
            module (`torch.nn.Module`):
                The module in which the tensor we want to move lives.
            tensor_name (`str`):
                The full name of the parameter/buffer.
            device (`int`, `str` or `torch.device`):
                The device on which to set the tensor.
            value (`torch.Tensor`, *optional*):
                The value of the tensor (useful when going from the meta device to any other device).
            dtype (`torch.dtype`, *optional*):
                If passed along the value of the parameter will be cast to this `dtype`. Otherwise, `value` will be cast to
                the dtype of the existing parameter in the model.
            fp16_statistics (`torch.HalfTensor`, *optional*):
                The list of fp16 statistics to set on the module, used for 8 bit model serialization.
            tied_params_map (Dict[int, Dict[torch.device, torch.Tensor]], *optional*, defaults to `None`):
                A map of current data pointers to dictionaries of devices to already dispatched tied weights. For a given
                execution device, this parameter is useful to reuse the first available pointer of a shared weight on the
                device for all others, instead of duplicating memory.
        """
        # Recurse if needed
        if "." in tensor_name:
            splits = tensor_name.split(".")
            for split in splits[:-1]:
                new_module = getattr(module, split)
                if new_module is None:
                    raise ValueError(f"{module} has no attribute {split}.")
                module = new_module
            tensor_name = splits[-1]

        if tensor_name not in module._parameters and tensor_name not in module._buffers:
            raise ValueError(
                f"{module} does not have a parameter or a buffer named {tensor_name}."
            )
        is_buffer = tensor_name in module._buffers
        old_value = getattr(module, tensor_name)

        # Treat the case where old_value (or a custom `value`, typically offloaded to RAM/disk) belongs to a tied group, and one of the weight
        # in the tied group has already been dispatched to the device, by avoiding reallocating memory on the device and just copying the pointer.
        if (
            value is not None
            and tied_params_map is not None
            and value.data_ptr() in tied_params_map
            and device in tied_params_map[value.data_ptr()]
        ):
            module._parameters[tensor_name] = tied_params_map[value.data_ptr()][device]
            return
        elif (
            tied_params_map is not None
            and old_value.data_ptr() in tied_params_map
            and device in tied_params_map[old_value.data_ptr()]
        ):
            module._parameters[tensor_name] = tied_params_map[old_value.data_ptr()][
                device
            ]
            return

        if (
            old_value.device == torch.device("meta")
            and device not in ["meta", torch.device("meta")]
            and value is None
        ):
            raise ValueError(
                f"{tensor_name} is on the meta device, we need a `value` to put in on {device}."
            )

        if value is not None:
            if old_value.shape != value.shape:
                raise ValueError(
                    f'Trying to set a tensor of shape {value.shape} in "{tensor_name}" (which has shape {old_value.shape}), this look incorrect.'
                )

            if dtype is None:
                # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
                value = value.to(old_value.dtype)
            elif not str(value.dtype).startswith(
                ("torch.uint", "torch.int", "torch.bool")
            ):
                value = value.to(dtype)

        param = (
            module._parameters[tensor_name]
            if tensor_name in module._parameters
            else None
        )
        param_cls = type(param)

        device_quantization = None
        with torch.no_grad():
            # leave it on cpu first before moving them to cuda
            # # fix the case where the device is meta, we don't want to put it on cpu because there is no data =0
            if (
                param is not None
                and param.device.type != "cuda"
                and torch.device(device).type == "cuda"
                and param_cls.__name__ in ["Int8Params", "FP4Params", "Params4bit"]
            ):
                device_quantization = device
                device = "cpu"
            # `torch.Tensor.to(<int num>)` is not supported by `torch_npu` (see this [issue](https://github.com/Ascend/pytorch/issues/16)).
            if is_npu_available() and isinstance(device, int):
                device = f"npu:{device}"
            if is_xpu_available() and isinstance(device, int):
                device = f"xpu:{device}"
            if value is None:
                new_value = old_value.to(device)
                if dtype is not None and device in ["meta", torch.device("meta")]:
                    if not str(old_value.dtype).startswith(
                        ("torch.uint", "torch.int", "torch.bool")
                    ):
                        new_value = new_value.to(dtype)

                    if not is_buffer:
                        module._parameters[tensor_name] = param_cls(
                            new_value, requires_grad=old_value.requires_grad
                        )
            elif isinstance(value, torch.Tensor):
                new_value = value.to(device)
            else:
                new_value = torch.tensor(value, device=device)
            if device_quantization is not None:
                device = device_quantization
            if is_buffer:
                module._buffers[tensor_name] = new_value
            elif value is not None or not check_device_same(
                torch.device(device), module._parameters[tensor_name].device
            ):
                param_cls = type(module._parameters[tensor_name])
                kwargs = module._parameters[tensor_name].__dict__
                if param_cls.__name__ in ["Int8Params", "FP4Params"]:
                    if (
                        param_cls.__name__ == "Int8Params"
                        and new_value.dtype == torch.float32
                    ):
                        # downcast to fp16 if any - needed for 8bit serialization
                        new_value = new_value.to(torch.float16)
                    # quantize module that are going to stay on the cpu so that we offload quantized weights
                    if device == "cpu" and param_cls.__name__ == "Int8Params":
                        new_value = (
                            param_cls(
                                new_value,
                                requires_grad=old_value.requires_grad,
                                **kwargs,
                            )
                            .to(0)
                            .to("cpu")
                        )
                        new_value.CB = new_value.CB.to("cpu")
                        new_value.SCB = new_value.SCB.to("cpu")
                    else:
                        new_value = param_cls(
                            new_value, requires_grad=old_value.requires_grad, **kwargs
                        ).to(device)
                elif param_cls.__name__ in ["QTensor", "QBitsTensor"]:
                    new_value = torch.nn.Parameter(
                        new_value, requires_grad=old_value.requires_grad
                    ).to(device)
                elif isinstance(new_value, FakeTensor) or isinstance(
                    old_value, FakeTensor
                ):
                    new_value = torch.nn.Parameter(
                        new_value, requires_grad=old_value.requires_grad
                    ).to(device)
                else:
                    new_value = param_cls(
                        new_value, requires_grad=old_value.requires_grad
                    ).to(device)

                module._parameters[tensor_name] = new_value
                if fp16_statistics is not None:
                    module._parameters[tensor_name].SCB = fp16_statistics.to(device)
                    del fp16_statistics
                # as we put the weight to meta, it doesn't have SCB attr anymore. make sure that it is not a meta weight
                if (
                    module.__class__.__name__ == "Linear8bitLt"
                    and getattr(module.weight, "SCB", None) is None
                    and str(module.weight.device) != "meta"
                ):
                    # quantize only if necessary
                    device_index = (
                        torch.device(device).index
                        if torch.device(device).type == "cuda"
                        else None
                    )
                    if (
                        not getattr(module.weight, "SCB", None)
                        and device_index is not None
                    ):
                        if (
                            module.bias is not None
                            and module.bias.device.type != "meta"
                        ):
                            # if a bias exists, we need to wait until the bias is set on the correct device
                            module = module.cuda(device_index)
                        elif module.bias is None:
                            # if no bias exists, we can quantize right away
                            module = module.cuda(device_index)
                elif (
                    module.__class__.__name__ == "Linear4bit"
                    and getattr(module.weight, "quant_state", None) is None
                ):
                    # quantize only if necessary
                    device_index = (
                        torch.device(device).index
                        if torch.device(device).type == "cuda"
                        else None
                    )
                    if (
                        not getattr(module.weight, "quant_state", None)
                        and device_index is not None
                    ):
                        module.weight = module.weight.cuda(device_index)
        # clean pre and post foward hook
        if is_npu_available():
            torch.npu.empty_cache()
        elif is_xpu_available():
            torch.xpu.empty_cache()
        else:
            torch.cuda.empty_cache()

        # When handling tied weights, we update tied_params_map to keep track of the tied weights that have already been allocated on the device in
        # order to avoid duplicating memory, see above.
        if (
            tied_params_map is not None
            and old_value.data_ptr() in tied_params_map
            and device not in tied_params_map[old_value.data_ptr()]
        ):
            tied_params_map[old_value.data_ptr()][device] = new_value
        elif (
            value is not None
            and tied_params_map is not None
            and value.data_ptr() in tied_params_map
            and device not in tied_params_map[value.data_ptr()]
        ):
            tied_params_map[value.data_ptr()][device] = new_value

    from accelerate import hooks

    DEFAULT_PATCHER.add(
        Patch(hooks, set_module_tensor_to_device, "set_module_tensor_to_device")
    )

except:
    pass


DEFAULT_PATCHER.__enter__()
