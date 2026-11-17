"""Minimal LoRA implementation (no external PEFT dependency).

Wraps existing nn.Linear layers with a low-rank additive branch:
    y = W x + (alpha / r) * B (A x)
The frozen base weight is never updated; only A and B are trained.
"""
import math

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scaling * self.dropout(x) @ self.lora_A.T @ self.lora_B.T

    @torch.no_grad()
    def merged_weight(self):
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)


def apply_lora(model: nn.Module, targets=DEFAULT_TARGETS, r=16, alpha=32, dropout=0.0):
    """Replace matching nn.Linear children in-place; return trainable parameter count."""
    for name, module in model.named_modules():
        for child_name in list(module._modules.keys()):
            child = module._modules[child_name]
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and full.split(".")[-1] in targets:
                module._modules[child_name] = LoRALinear(child, r, alpha, dropout)
    for p in model.parameters():
        p.requires_grad = False
    n = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
            n += p.numel()
    return n


def lora_state_dict(model: nn.Module):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


@torch.no_grad()
def merge_lora(model: nn.Module):
    """Merge LoRA branches into base weights and restore plain Linear modules."""
    for name, module in model.named_modules():
        for child_name in list(module._modules.keys()):
            child = module._modules[child_name]
            if isinstance(child, LoRALinear):
                merged = nn.Linear(child.base.in_features, child.base.out_features,
                                   bias=child.base.bias is not None)
                merged.weight.copy_(child.merged_weight())
                if child.base.bias is not None:
                    merged.bias.copy_(child.base.bias)
                module._modules[child_name] = merged
    return model
