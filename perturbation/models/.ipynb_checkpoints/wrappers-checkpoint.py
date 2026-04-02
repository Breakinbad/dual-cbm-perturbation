from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class PerturbationModel(nn.Module):
    """Thin wrapper around your scGPT backbone.

    Update the forward call here so the rest of the codebase does not depend on
    notebook-specific conventions.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        gene_ids: torch.Tensor,
        values: torch.Tensor,
        drug_emb: torch.Tensor,
        task_ids: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.backbone(
            gene_ids,
            values,
            drug_emb,
            task_ids=task_ids,
            src_key_padding_mask=src_key_padding_mask,
        )


def freeze_all_parameters(model: nn.Module) -> None:
    for _, p in model.named_parameters():
        p.requires_grad = False


def unfreeze_modules_by_substring(model: nn.Module, substrings: list[str]) -> list[str]:
    train_names: list[str] = []
    for name, p in model.named_parameters():
        if any(token in name for token in substrings):
            p.requires_grad = True
            train_names.append(name)
    return train_names


def build_optimizer(model: nn.Module, lr: float = 5e-4, weight_decay: float = 1e-3):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bias_or_norm = (p.ndim == 1) or name.endswith(".bias") or ("norm" in name.lower())
        is_gate = name.endswith(".alpha") or ("alpha" in name)
        (no_decay if (is_bias_or_norm or is_gate) else decay).append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
