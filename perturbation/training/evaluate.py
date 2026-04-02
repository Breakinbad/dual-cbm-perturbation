from __future__ import annotations

import torch
import torch.nn.functional as F

from .metrics import pearsonr_batch


def _safe_accuracy(
    logits: torch.Tensor | None,
    targets: torch.Tensor,
    sample_mask: torch.Tensor,
) -> tuple[float, float, int]:
    """Compute masked CE loss and accuracy for sample-level classification heads."""
    if logits is None or sample_mask.sum().item() == 0:
        return 0.0, 0.0, 0

    logits = logits[sample_mask]
    targets = targets[sample_mask]

    loss = F.cross_entropy(logits, targets).item()
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return loss, correct, total


@torch.no_grad()
def validate_four_tasks(
    model,
    val_loader,
    device: torch.device,
    use_bf16: bool = True,
    loss_weights: dict[str, float] | None = None,
):
    model.eval()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    loss_weights = loss_weights or {
        "expr": 1.0,
        "direction": 1.0,
        "go": 1.0,
        "moa_broad": 1.0,
    }

    expr_total = 0.0
    expr_r = 0.0
    expr_count = 0

    direction_loss = 0.0
    direction_correct = 0.0
    direction_total = 0.0

    go_loss = 0.0
    go_correct = 0.0
    go_total = 0.0

    moa_loss = 0.0
    moa_correct = 0.0
    moa_total = 0.0

    for batch in val_loader:
        gene_ids = batch["gene_ids"].long().to(device)
        control_expr = batch["values"].float().to(device)
        drug_emb = batch["drug_emb"].float().to(device)
        target_expr = batch["target_expr"].float().to(device)
        target_direction = batch["target_direction"].long().to(device)
        target_go = batch["target_go"].long().to(device)
        target_moa = batch["target_moa_broad"].long().to(device)
        go_mask = batch["go_mask"].bool().to(device)
        moa_mask = batch["moa_mask"].bool().to(device)
        control_std = batch["control_std"].float().to(device)
        padding_mask = batch["padding_mask"].to(device)

        eps = control_std.mean(dim=1, keepdim=True).clamp(min=1e-6)
        denom = control_std + eps

        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=torch.cuda.is_available(),
        ):
            out = model(
                gene_ids,
                control_expr,
                drug_emb,
                src_key_padding_mask=padding_mask,
            )

            # Expression
            expr_pred = out["mlm_output"].float()
            d_pred = (expr_pred - control_expr) / denom
            d_true = (target_expr - control_expr) / denom
            expr_mse = F.mse_loss(d_pred, d_true)

            expr_total += expr_mse.item() * gene_ids.size(0)
            expr_r += pearsonr_batch(d_true, d_pred).sum().item()
            expr_count += gene_ids.size(0)

            # Direction
            if "direction_logits" in out:
                logits = out["direction_logits"].float().reshape(-1, 3)
                flat_targets = target_direction.reshape(-1)

                direction_loss_batch = F.cross_entropy(logits, flat_targets)
                direction_preds = logits.argmax(dim=-1)

                direction_correct += (direction_preds == flat_targets).sum().item()
                direction_total += flat_targets.numel()
                direction_loss += direction_loss_batch.item() * flat_targets.numel()

            # GO
            go_loss_batch, go_correct_batch, go_total_batch = _safe_accuracy(
                out.get("go_logits"),
                target_go,
                go_mask,
            )
            go_loss += go_loss_batch * go_total_batch
            go_correct += go_correct_batch
            go_total += go_total_batch

            # MOA broad
            moa_loss_batch, moa_correct_batch, moa_total_batch = _safe_accuracy(
                out.get("moa_broad_logits"),
                target_moa,
                moa_mask,
            )
            moa_loss += moa_loss_batch * moa_total_batch
            moa_correct += moa_correct_batch
            moa_total += moa_total_batch

    expr_avg = expr_total / max(expr_count, 1)
    direction_avg = direction_loss / max(direction_total, 1)
    go_avg = go_loss / max(go_total, 1)
    moa_avg = moa_loss / max(moa_total, 1)

    monitor_total = (
        loss_weights["expr"] * expr_avg
        + loss_weights["direction"] * direction_avg
        + loss_weights["go"] * go_avg
        + loss_weights["moa_broad"] * moa_avg
    )

    return {
        "total": monitor_total,
        "expr": {
            "total": expr_avg,
            "r": expr_r / max(expr_count, 1),
            "mse": expr_avg,
        },
        "direction": {
            "total": direction_avg,
            "accuracy": direction_correct / max(direction_total, 1),
        },
        "go": {
            "total": go_avg,
            "accuracy": go_correct / max(go_total, 1),
        },
        "moa_broad": {
            "total": moa_avg,
            "accuracy": moa_correct / max(moa_total, 1),
        },
    }


@torch.no_grad()
def evaluate_deltas_vs_cellmean(
    model,
    loader,
    device: torch.device,
    use_bf16: bool = True,
) -> dict[str, float]:
    model.eval()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    sums = {
        "model_r": 0.0,
        "model_mse": 0.0,
        "base_r": 0.0,
        "base_mse": 0.0,
    }
    count = 0

    for batch in loader:
        ctrl = batch["values"].float().to(device)
        tgt = batch["target_expr"].float().to(device)
        drug_emb = batch["drug_emb"].float().to(device)
        g_ids = batch["gene_ids"].long().to(device)
        ctrl_std = batch["control_std"].float().to(device)
        avg_train = batch["avg_exp_train"].float().to(device)
        padding_mask = batch["padding_mask"].to(device)

        eps = ctrl_std.mean(dim=1, keepdim=True).clamp(min=1e-6)
        denom = ctrl_std + eps

        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=torch.cuda.is_available(),
        ):
            out = model(
                g_ids,
                ctrl,
                drug_emb,
                src_key_padding_mask=padding_mask,
            )
            expr_pred = out["mlm_output"].float()

            d_true = (tgt - ctrl) / denom
            d_model = (expr_pred - ctrl) / denom
            d_base = (avg_train - ctrl) / denom

        batch_size = ctrl.size(0)
        sums["model_r"] += pearsonr_batch(d_true, d_model).sum().item()
        sums["base_r"] += pearsonr_batch(d_true, d_base).sum().item()
        sums["model_mse"] += F.mse_loss(d_model, d_true, reduction="sum").item()
        sums["base_mse"] += F.mse_loss(d_base, d_true, reduction="sum").item()
        count += batch_size

    return {
        "model_r": sums["model_r"] / max(count, 1),
        "base_r": sums["base_r"] / max(count, 1),
        "model_mse": sums["model_mse"] / max(count, 1),
        "base_mse": sums["base_mse"] / max(count, 1),
    }