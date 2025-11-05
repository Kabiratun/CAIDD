import math
import torch
import torch.nn as nn

class LoRAQKV(nn.Module):
    """Wrap a 1×1 Conv1d used for QKV projection with a low-rank adapter."""
    def __init__(self, base_conv: nn.Conv1d, r: int = 8, alpha: int = 8):
        super().__init__()
        if not isinstance(base_conv, nn.Conv1d):
            raise TypeError("LoRAQKV expects nn.Conv1d")
        self.base = base_conv
        self.r = r
        self.scaling = alpha / r
        in_c, out_c = base_conv.in_channels, base_conv.out_channels
        self.A = nn.Conv1d(in_c, r, 1, bias=False)
        self.B = nn.Conv1d(r, out_c, 1, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.scaling


def mark_only_lora_trainable(module: nn.Module):
    """Freeze all base parameters; leave LoRA A/B trainable."""
    for p in module.parameters():
        p.requires_grad = False
    for m in module.modules():
        if isinstance(m, LoRAQKV):
            for p in list(m.A.parameters()) + list(m.B.parameters()):
                p.requires_grad = True
    return module

