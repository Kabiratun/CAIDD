# patch_lora.py  (place at project root, e.g., /content/CAIDD/patch_lora.py)
import torch.nn as nn

try:
    # must exist in your repo root as lora.py (you said you created it)
    from lora import LoRAQKV
except Exception as e:
    raise ImportError("Could not import LoRAQKV from lora.py. Make sure lora.py is in the project root and defines class LoRAQKV.") from e


def _get_parent_and_attr(model: nn.Module, dotted_name: str):
    """
    Return (parent_module, final_attr_name) so we can replace parent.final_attr_name.
    Example: dotted_name='blocks.3.attn.qkv' -> returns (model.blocks[3].attn, 'qkv')
    """
    parts = dotted_name.split(".")
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]


def enable_lora_on_qkv(model: nn.Module, rank: int = 8, alpha: int = 8, verbose: bool = True):
    """
    Wrap every Conv1d module named ...'.qkv' with a LoRAQKV adapter.
    This makes no assumptions about your model classes/paths.
    """
    wrapped = 0
    for name, module in list(model.named_modules()):
        # We only target 1x1 Conv1d typical in attention blocks for QKV
        is_qkv_name = name.endswith(".qkv") or name.split(".")[-1] == "qkv"
        if is_qkv_name and isinstance(module, nn.Conv1d):
            parent, attr = _get_parent_and_attr(model, name)
            # Skip if already wrapped
            if isinstance(getattr(parent, attr), LoRAQKV):
                continue
            setattr(parent, attr, LoRAQKV(module, r=rank, alpha=alpha))
            wrapped += 1
    if verbose:
        print(f"[LoRA] wrapped {wrapped} attention projection(s) named 'qkv'.")
    return wrapped
