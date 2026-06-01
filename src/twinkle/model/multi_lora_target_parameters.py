from __future__ import annotations

import math
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator

import torch
from peft import LoraConfig
from torch import nn


class LoraParameterProxy(nn.Module):

    def __init__(self, delta_weight: torch.Tensor):
        super().__init__()
        self.delta_weight = delta_weight

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight + self.delta_weight.to(device=weight.device, dtype=weight.dtype)


@dataclass
class TargetParameterRecord:
    module_name: str
    module: nn.Module
    parameter_name: str

    @property
    def key(self) -> str:
        if self.module_name:
            return f'{self.module_name}.{self.parameter_name}'
        return self.parameter_name


class TargetParameterLoraWrapper(nn.Module):

    def __init__(self, record: TargetParameterRecord, max_loras: int, max_r: int):
        super().__init__()
        self.record = record
        self.max_loras = max_loras
        self.max_r = max_r
        self.active_adapter: str | None = None
        self.disable_adapters = False
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.scaling: dict[str, float] = {}
        self.r: dict[str, int] = {}
        self._init_slots()

    @property
    def base_parameter(self) -> nn.Parameter:
        return getattr(self.record.module, self.record.parameter_name)

    @property
    def num_experts(self) -> int:
        parameter = self.base_parameter
        return parameter.shape[0] if parameter.ndim == 3 else 1

    @property
    def in_features(self) -> int:
        parameter = self.base_parameter
        return parameter.shape[-2] if parameter.ndim == 3 else parameter.shape[1]

    @property
    def out_features(self) -> int:
        parameter = self.base_parameter
        return parameter.shape[-1] if parameter.ndim == 3 else parameter.shape[0]

    def _init_slots(self) -> None:
        parameter = self.base_parameter
        if parameter.ndim not in (2, 3):
            raise ValueError(
                f'target parameter {self.record.key} has {parameter.ndim} dimensions; only 2D and 3D are supported')

        for index in range(self.max_loras):
            slot_name = f'lora_{index}'
            self.lora_A[slot_name] = nn.Linear(
                self.in_features,
                self.max_r * self.num_experts,
                bias=False,
                device=parameter.device,
                dtype=parameter.dtype,
            )
            self.lora_B[slot_name] = nn.Linear(
                self.max_r * self.num_experts,
                self.out_features,
                bias=False,
                device=parameter.device,
                dtype=parameter.dtype,
            )
            self.r[slot_name] = self.max_r
            self.scaling[slot_name] = 1.0
            self.reset_slot(slot_name)

    def reset_slot(self, slot_name: str) -> None:
        nn.init.kaiming_uniform_(self.lora_A[slot_name].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B[slot_name].weight)

    def configure_slot(self, slot_name: str, config: LoraConfig) -> None:
        if slot_name not in self.lora_A:
            raise ValueError(f'Unknown target-parameter LoRA slot: {slot_name}')
        if config.r <= 0:
            raise ValueError(f'`r` should be a positive integer value but the value passed is {config.r}')
        if config.r > self.max_r:
            raise ValueError(f'LoRA rank {config.r} exceeds max_r={self.max_r}')
        if getattr(config, 'lora_dropout', 0):
            raise ValueError('target_parameters LoRA does not support lora_dropout != 0')
        if getattr(config, 'fan_in_fan_out', False):
            raise ValueError('target_parameters LoRA does not support fan_in_fan_out=True')
        if getattr(config, 'use_dora', False):
            raise ValueError('target_parameters LoRA does not support use_dora=True')
        if getattr(config, 'lora_bias', False):
            raise ValueError('target_parameters LoRA does not support lora_bias=True')

        self.r[slot_name] = config.r
        if getattr(config, 'use_rslora', False):
            self.scaling[slot_name] = config.lora_alpha / math.sqrt(config.r)
        else:
            self.scaling[slot_name] = config.lora_alpha / config.r

    def get_delta_weight(self, slot_name: str) -> torch.Tensor:
        if slot_name not in self.lora_A:
            raise ValueError(f'Unknown target-parameter LoRA slot: {slot_name}')

        r = self.r[slot_name]
        num_experts = self.num_experts
        weight_A = self.lora_A[slot_name].weight[:r * num_experts, :]
        weight_B = self.lora_B[slot_name].weight[:, :r * num_experts]

        if self.base_parameter.ndim == 2:
            return (weight_B @ weight_A) * self.scaling[slot_name]

        weight_A = weight_A.reshape(num_experts, r, self.in_features)
        weight_B = weight_B.reshape(self.out_features, r, num_experts)
        return torch.einsum('o r e, e r i -> e i o', weight_B, weight_A) * self.scaling[slot_name]

    @contextmanager
    def activate(self, slot_name: str | None, disable_lora: bool = False):
        if disable_lora or slot_name is None or slot_name not in self.lora_A:
            yield
            return

        delta_weight = self.get_delta_weight(slot_name)
        requires_grad_before = self.base_parameter.requires_grad
        nn.utils.parametrize.register_parametrization(
            self.record.module,
            self.record.parameter_name,
            LoraParameterProxy(delta_weight),
        )
        self.record.module.parametrizations[self.record.parameter_name].original.requires_grad_(requires_grad_before)
        try:
            with nn.utils.parametrize.cached():
                yield
        finally:
            nn.utils.parametrize.remove_parametrizations(
                self.record.module,
                self.record.parameter_name,
                leave_parametrized=False,
            )

    def named_slot_parameters(self, slot_name: str) -> Iterator[tuple[str, nn.Parameter]]:
        if slot_name not in self.lora_A:
            return
        yield f'{self.record.key}.lora_A.{slot_name}.weight', self.lora_A[slot_name].weight
        yield f'{self.record.key}.lora_B.{slot_name}.weight', self.lora_B[slot_name].weight

    def parameters_for_slot(self, slot_name: str) -> list[nn.Parameter]:
        return [parameter for _, parameter in self.named_slot_parameters(slot_name)]


class TargetParameterLoraManager:

    def __init__(self, max_loras: int, max_r: int):
        self.max_loras = max_loras
        self.max_r = max_r
        self.wrappers: list[TargetParameterLoraWrapper] = []
        self.tenant_to_slot: dict[str, str] = {}
        self.tenant_configs: dict[str, LoraConfig] = {}
        self._target_parameters: tuple[str, ...] | None = None

    def patch(self, model: nn.Module, target_parameters: Iterable[str]) -> None:
        target_parameters = tuple(target_parameters)
        if not target_parameters:
            return
        if self._target_parameters is not None:
            if self._target_parameters != target_parameters:
                raise ValueError(
                    f'target_parameters already patched as {self._target_parameters}, got {target_parameters}')
            return

        records = []
        for module_name, module in model.named_modules():
            for param_name, parameter in module.named_parameters(recurse=False):
                key = f'{module_name}.{param_name}' if module_name else param_name
                if key in target_parameters or any(key.endswith(f'.{target}') for target in target_parameters):
                    if parameter.ndim not in (2, 3):
                        raise ValueError(
                            f'target parameter {key} has {parameter.ndim} dimensions; only 2D and 3D are supported')
                    records.append(TargetParameterRecord(module_name, module, param_name))

        if not records:
            raise ValueError(f'target_parameters={target_parameters} were set but no parameter was matched')

        for record in records:
            wrapper = TargetParameterLoraWrapper(record, max_loras=self.max_loras, max_r=self.max_r)
            record.module.add_module(f'_twinkle_lora_{record.parameter_name}', wrapper)
            self.wrappers.append(wrapper)
        self._target_parameters = target_parameters

    def acquire(self, tenant_adapter_name: str, slot_name: str, config: LoraConfig) -> None:
        if tenant_adapter_name in self.tenant_to_slot:
            raise ValueError(f'Lora {tenant_adapter_name} already exists')
        if getattr(config, 'target_parameters', None) and not self.wrappers:
            raise ValueError('target_parameters LoRA slots must be patched before acquire')

        self.tenant_to_slot[tenant_adapter_name] = slot_name
        self.tenant_configs[tenant_adapter_name] = config
        for wrapper in self.wrappers:
            wrapper.configure_slot(slot_name, config)

    def release(self, tenant_adapter_name: str) -> None:
        slot_name = self.tenant_to_slot.pop(tenant_adapter_name, None)
        self.tenant_configs.pop(tenant_adapter_name, None)
        if slot_name is None:
            return
        for wrapper in self.wrappers:
            wrapper.reset_slot(slot_name)

    @contextmanager
    def adapter(self, tenant_adapter_name: str, disable_lora: bool = False):
        slot_name = self.tenant_to_slot.get(tenant_adapter_name)
        with ExitStack() as stack:
            for wrapper in self.wrappers:
                stack.enter_context(wrapper.activate(slot_name, disable_lora=disable_lora))
            yield

    def parameters_for_tenant(self, tenant_adapter_name: str) -> list[nn.Parameter]:
        slot_name = self.tenant_to_slot[tenant_adapter_name]
        parameters = []
        for wrapper in self.wrappers:
            parameters.extend(wrapper.parameters_for_slot(slot_name))
        return parameters

    def named_slot_parameters(self, tenant_adapter_name: str) -> Iterator[tuple[str, nn.Parameter]]:
        slot_name = self.tenant_to_slot[tenant_adapter_name]
        for wrapper in self.wrappers:
            yield from wrapper.named_slot_parameters(slot_name)
