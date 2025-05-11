# claude_ilora.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class iLoRALayer(nn.Module):
    """
    Low-Rank Adaptation (iLoRA) adapter layer.
    """
    def __init__(self, in_features, out_features, rank=8, alpha=32, dropout=0.0, use_gating=True):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.use_gating = use_gating

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros((rank, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))
        if use_gating:
            self.lora_gate = nn.Parameter(torch.zeros(1))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        if self.use_gating:
            nn.init.zeros_(self.lora_gate)

    def forward(self, x):
        x_d = self.lora_dropout(x)
        inter = F.linear(x_d, self.lora_A)          # [*, in] -> [*, rank]
        lora_out = F.linear(inter, self.lora_B) * self.scaling  # [*, rank] -> [*, out]
        if self.use_gating:
            lora_out = lora_out * torch.sigmoid(self.lora_gate)
        return lora_out


class iLoRALinear(nn.Module):
    """
    Wraps nn.Linear to add a frozen base + LoRA adapter.
    """
    def __init__(self, base_linear: nn.Linear, rank=8, alpha=32, dropout=0.0, use_gating=True):
        super().__init__()
        # freeze base
        self.base_layer = base_linear
        for p in self.base_layer.parameters():
            p.requires_grad = False
        # LoRA adapter
        self.lora_layer = iLoRALayer(
            in_features=base_linear.in_features,
            out_features=base_linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            use_gating=use_gating,
        )
        # expose weight/bias attributes
        self.weight = self.base_layer.weight
        self.bias   = self.base_layer.bias

    def forward(self, x):
        return self.base_layer(x) + self.lora_layer(x)


def replace_with_ilora(
    model: nn.Module,
    target_modules=None,
    rank=8,
    alpha=32,
    dropout=0.0,
    use_gating=True,
):
    """
    Recursively replace Linear modules with iLoRALinear, then freeze all non-LoRA params.
    """
    if target_modules is None:
        target_modules = [nn.Linear]
    for name, child in list(model.named_children()):
        if any(isinstance(child, tm) for tm in target_modules):
            wrapped = iLoRALinear(
                base_linear=child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                use_gating=use_gating,
            )
            setattr(model, name, wrapped)
        else:
            replace_with_ilora(
                child,
                target_modules=target_modules,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                use_gating=use_gating,
            )
    # freeze everything except LoRA
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    return model


def calculate_ilora_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total,
        'trainable_params': trainable,
        'trainable_percentage': 100.0 * trainable / total,
    }