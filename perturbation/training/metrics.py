from __future__ import annotations

import torch


def compute_delta(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    ctrl: torch.Tensor,
    ctrl_std: torch.Tensor | None = None,
    use_norm: bool = True,
):
    d_pred = pred - ctrl
    d_true = tgt - ctrl
    if use_norm and ctrl_std is not None:
        eps = ctrl_std.mean(dim=1, keepdim=True).clamp(min=1e-6)
        denom = ctrl_std + eps
        d_pred = d_pred / denom
        d_true = d_true / denom
    return d_pred, d_true


def pearsonr_batch(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    y_true = y_true - y_true.mean(dim=1, keepdim=True)
    y_pred = y_pred - y_pred.mean(dim=1, keepdim=True)
    num = (y_true * y_pred).sum(dim=1)
    den = torch.sqrt((y_true.square().sum(dim=1) + 1e-8) * (y_pred.square().sum(dim=1) + 1e-8))
    return num / den
