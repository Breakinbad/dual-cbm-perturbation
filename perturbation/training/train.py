from __future__ import annotations

import torch
import torch.nn.functional as F

from .evaluate import validate_four_tasks
from perturbation.training.losses import FourTaskLoss

def _masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    if logits is None or mask.sum().item() == 0:
        return torch.tensor(0.0, device=device)
    return F.cross_entropy(logits[mask], targets[mask])


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device: torch.device,
    epoch: int,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    use_bf16: bool = True,
) -> dict[str, float]:
    model.train()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    total_loss_sum = expr_sum = dir_sum = go_sum = moa_sum = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader, start=1):
        gene_ids = batch["gene_ids"].long().to(device)
        control_expr = batch["values"].float().to(device)
        drug_emb = batch["drug_emb"].float().to(device)
        padding_mask = batch["padding_mask"].to(device)

        targets = {
            "expr": batch["target_expr"].float().to(device),
            "direction": batch["target_direction"].long().to(device),
            "go": batch["target_go"].long().to(device),
            "moa_broad": batch["target_moa_broad"].long().to(device),
        }

        masks = {
            "go": batch["go_mask"].bool().to(device),
            "moa_broad": batch["moa_mask"].bool().to(device),
        }

        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=torch.cuda.is_available(),
        ):
            predictions = model(
                gene_ids,
                control_expr,
                drug_emb,
                src_key_padding_mask=padding_mask,
            )

            total_loss, task_losses = criterion(
                predictions=predictions,
                targets=targets,
                masks=masks,
            )
            loss = total_loss / grad_accum_steps

        loss.backward()

        if step % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss_sum += float(total_loss.item())
        expr_sum += task_losses["expr"]
        dir_sum += task_losses["direction"]
        go_sum += task_losses["go"]
        moa_sum += task_losses["moa_broad"]
        n_batches += 1

    return {
        "total": total_loss_sum / max(n_batches, 1),
        "expr": expr_sum / max(n_batches, 1),
        "direction": dir_sum / max(n_batches, 1),
        "go": go_sum / max(n_batches, 1),
        "moa_broad": moa_sum / max(n_batches, 1),
    }





def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    device: torch.device,
    epochs: int = 20,
    loss_weights: dict[str, float] | None = None,
) -> dict[str, list[float]]:
    loss_weights = loss_weights or {
        "expr": 1.0,
        "direction": 1.0,
        "go": 1.0,
        "moa_broad": 1.0,
    }

    criterion = FourTaskLoss(
        expr_weight=loss_weights["expr"],
        direction_weight=loss_weights["direction"],
        go_weight=loss_weights["go"],
        moa_weight=loss_weights["moa_broad"],
        direction_class_weights=torch.tensor([2.0, 0.5, 2.0], device=device),
    )

    history = {
        "train_total": [],
        "val_total": [],
        "train_expr": [],
        "train_direction": [],
        "train_go": [],
        "train_moa_broad": [],
        "val_expr": [],
        "val_direction": [],
        "val_go": [],
        "val_moa_broad": [],
    }

    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        val_stats = validate_four_tasks(
            model=model,
            val_loader=val_loader,
            device=device,
            loss_weights=loss_weights,
        )

        history["train_total"].append(train_stats["total"])
        history["val_total"].append(val_stats["total"])
        history["train_expr"].append(train_stats["expr"])
        history["train_direction"].append(train_stats["direction"])
        history["train_go"].append(train_stats["go"])
        history["train_moa_broad"].append(train_stats["moa_broad"])
        history["val_expr"].append(val_stats["expr"]["total"])
        history["val_direction"].append(val_stats["direction"]["total"])
        history["val_go"].append(val_stats["go"]["total"])
        history["val_moa_broad"].append(val_stats["moa_broad"]["total"])

        if val_stats["total"] < best_val:
            best_val = val_stats["total"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        print(
            f"epoch={epoch:03d} "
            f"train_total={train_stats['total']:.4f} "
            f"val_total={val_stats['total']:.4f} "
            f"val_expr_r={val_stats['expr']['r']:.4f} "
            f"val_dir_acc={val_stats['direction']['accuracy']:.4f} "
            f"val_go_acc={val_stats['go']['accuracy']:.4f} "
            f"val_moa_acc={val_stats['moa_broad']['accuracy']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history