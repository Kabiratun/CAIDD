
import math
import torch
import torch.nn as nn

class LoRAQKV(nn.Module):
    """
    Wrap a 1x1 conv/linear (Q/K/V). Keeps base frozen; learns low-rank A,B.
    """
    def __init__(self, base_layer: nn.Module, r: int = 8, alpha: int = 8, dropout_p: float = 0.0):
        super().__init__()
        self.base = base_layer
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        if isinstance(base_layer, nn.Conv1d):
            out_c, in_c, k = base_layer.weight.shape
            assert k == 1, "LoRAQKV expects Conv1d with kernel_size=1"
            self.A = nn.Conv1d(in_c, r, kernel_size=1, bias=False)
            self.B = nn.Conv1d(r, out_c, kernel_size=1, bias=False)
        elif isinstance(base_layer, nn.Conv2d):
            out_c, in_c, kh, kw = base_layer.weight.shape
            assert kh == 1 and kw == 1, "LoRAQKV expects Conv2d 1x1"
            self.A = nn.Conv2d(in_c, r, kernel_size=1, bias=False)
            self.B = nn.Conv2d(r, out_c, kernel_size=1, bias=False)
        elif isinstance(base_layer, nn.Linear):
            out_f, in_f = base_layer.weight.shape
            self.A = nn.Linear(in_f, r, bias=False)
            self.B = nn.Linear(r, out_f, bias=False)
        else:
            raise TypeError(f"Unsupported base layer type: {type(base_layer)}")

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.B(self.A(self.dropout(x))) * self.scaling
        return base_out + lora_out

def mark_only_lora_trainable(model: nn.Module):
    trainable = 0
    total = 0
    for n, p in model.named_parameters():
        total += p.numel()
        if "A.weight" in n or "B.weight" in n:
            p.requires_grad_(True)
            trainable += p.numel()
        else:
            p.requires_grad_(False)
    print(f"[LoRA] trainable params: {trainable:,} / total: {total:,} "
          f"({100*trainable/total:.2f}%)")
