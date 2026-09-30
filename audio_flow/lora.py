from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Minimal LoRA wrapper that keeps the original Linear layer frozen."""

    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.linear = linear
        self.scale = alpha / rank
        self.lora_a = nn.Linear(linear.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        # LoRA can be injected after the base model has already been moved to
        # CUDA, so the newly-created parameters must follow the wrapped layer.
        self.lora_a.to(device=linear.weight.device, dtype=linear.weight.dtype)
        self.lora_b.to(device=linear.weight.device, dtype=linear.weight.dtype)
        for parameter in self.linear.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora_b(self.lora_a(x)) * self.scale


def inject_lora(
    module: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    targets: tuple[str, ...] = ("qkv_linear", "proj", "ffn.0", "ffn.2"),
) -> list[str]:
    """Replace matching Linear modules and return their full names."""

    replaced = []
    for name, child in list(module.named_modules()):
        if not isinstance(child, nn.Linear) or not name.endswith(targets):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha))
        replaced.append(name)
    if not replaced:
        raise ValueError(f"No Linear modules matched LoRA targets: {targets}")
    return replaced


def trainable_parameters(module: nn.Module):
    return [parameter for parameter in module.parameters() if parameter.requires_grad]
