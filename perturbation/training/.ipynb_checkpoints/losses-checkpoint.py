from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from scgpt.model.config import TASK_ID_TO_NAME, TASK_NAME_TO_ID, TASK_TYPES

def pearson_correlation_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    target_c = target - target.mean(dim=1, keepdim=True)
    cov = (pred_c * target_c).sum(dim=1)
    pred_std = torch.sqrt((pred_c.square()).sum(dim=1) + eps)
    target_std = torch.sqrt((target_c.square()).sum(dim=1) + eps)
    corr = cov / (pred_std * target_std)
    return 1.0 - corr.mean(), corr.mean().item()


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    ce_loss = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")
    p = torch.exp(-ce_loss)
    return (alpha * (1 - p) ** gamma * ce_loss).mean()


class FourTaskLoss(nn.Module):
    def __init__(
        self,
        expr_weight: float = 1.0,
        direction_weight: float = 1.0,
        go_weight: float = 1.0,
        moa_weight: float = 1.0,
        direction_class_weights: torch.Tensor | None = None,
        go_class_weights: torch.Tensor | None = None,
        moa_class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.expr_weight = expr_weight
        self.direction_weight = direction_weight
        self.go_weight = go_weight
        self.moa_weight = moa_weight

        self.direction_class_weights = direction_class_weights
        self.go_class_weights = go_class_weights
        self.moa_class_weights = moa_class_weights

    @staticmethod
    def _masked_cross_entropy(
        logits: torch.Tensor | None,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits is None:
            return targets.new_tensor(0.0, dtype=torch.float32)

        if mask is not None:
            if mask.sum().item() == 0:
                return targets.new_tensor(0.0, dtype=torch.float32)
            logits = logits[mask]
            targets = targets[mask]

        return F.cross_entropy(
            logits,
            targets.long(),
            weight=class_weights.to(logits.device) if class_weights is not None else None,
        )

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        masks = masks or {}

        expr_loss = F.mse_loss(predictions["mlm_output"], targets["expr"])

        dir_loss = self._masked_cross_entropy(
            predictions.get("direction_logits").reshape(-1, 3)
            if predictions.get("direction_logits") is not None
            else None,
            targets["direction"].reshape(-1),
            mask=None,
            class_weights=self.direction_class_weights,
        )

        go_loss = self._masked_cross_entropy(
            predictions.get("go_logits"),
            targets["go"],
            mask=masks.get("go"),
            class_weights=self.go_class_weights,
        )

        moa_loss = self._masked_cross_entropy(
            predictions.get("moa_broad_logits"),
            targets["moa_broad"],
            mask=masks.get("moa_broad"),
            class_weights=self.moa_class_weights,
        )

        total_loss = (
            self.expr_weight * expr_loss
            + self.direction_weight * dir_loss
            + self.go_weight * go_loss
            + self.moa_weight * moa_loss
        )

        task_losses = {
            "expr": float(expr_loss.item()),
            "direction": float(dir_loss.item()),
            "go": float(go_loss.item()),
            "moa_broad": float(moa_loss.item()),
        }
        return total_loss, task_losses

    