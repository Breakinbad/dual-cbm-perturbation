from __future__ import annotations

import argparse
from pathlib import Path

import torch

from perturbation.config import Config
from perturbation.data.dataset import DatasetContext, DropLastDataLoader, GeneExpressionDataset
from perturbation.data.io import load_anndata, load_pickle, load_torch_object
from perturbation.data.preprocessing import (
    build_cell_type_lookup,
    build_gene_index_maps,
    compute_average_expression,
    compute_control_statistics,
    compute_top_de_genes,
)
from perturbation.models.wrappers import PerturbationModel, build_optimizer
from perturbation.training.evaluate import evaluate_deltas_vs_cellmean
from perturbation.training.train import fit
from perturbation.utils.seed import seed_everything
from perturbation.training.losses import FourTaskLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def build_backbone_placeholder() -> torch.nn.Module:
    raise NotImplementedError(
        "Replace build_backbone_placeholder() with your actual scGPT model construction "
        "and checkpoint loading logic."
    )


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config).raw
    seed_everything(int(cfg.get("seed", 42)))

    adata = load_anndata(cfg["data"]["adata_path"])
    smiles_dict = load_torch_object(cfg["data"]["smiles_embeddings_path"])
    averaged_df_train = load_pickle(cfg["data"]["averaged_df_train_path"])
    averaged_df_test = load_pickle(cfg["data"]["averaged_df_test_path"])

    de_genes = compute_top_de_genes(
        adata,
        cell_type_key=cfg["preprocessing"]["cell_type_key"],
        condition_key=cfg["preprocessing"]["condition_key"],
        control_label=cfg["preprocessing"]["control_label"],
        top_k=int(cfg["preprocessing"]["de_top_k"]),
    )
    _, _, molecule_to_genes, _ = build_gene_index_maps(adata, de_genes)
    control_mean, control_std = compute_control_statistics(
        adata,
        cell_type_key=cfg["preprocessing"]["cell_type_key"],
        condition_key=cfg["preprocessing"]["condition_key"],
        control_label=cfg["preprocessing"]["control_label"],
    )
    average_expression = compute_average_expression(
        adata,
        cell_type_key=cfg["preprocessing"]["cell_type_key"],
        condition_key=cfg["preprocessing"]["condition_key"],
    )
    cell_types = sorted(adata.obs[cfg["preprocessing"]["cell_type_key"]].unique().tolist())
    cell_dic = build_cell_type_lookup(cell_types)
    average_expression_train = {
        cell_type: average_expression[(cell_type, cfg["preprocessing"]["control_label"])]
        for cell_type in cell_types
        if (cell_type, cfg["preprocessing"]["control_label"]) in average_expression
    }

    context = DatasetContext(
        molecule_to_genes=molecule_to_genes,
        average_expression=average_expression,
        control_data_dict=control_mean,
        control_std_dict=control_std,
        smiles_dict=smiles_dict,
        average_expression_train=average_expression_train,
        cell_dic=cell_dic,
        cell_types=cell_types,
    )

    train_ds = GeneExpressionDataset(
        averaged_df_train,
        split="train",
        context=context,
        threshold_magnitude=float(cfg["preprocessing"]["threshold_magnitude"]),
    )
    test_ds = GeneExpressionDataset(
        averaged_df_test,
        split="test",
        context=context,
        threshold_magnitude=float(cfg["preprocessing"]["threshold_magnitude"]),
    )

    train_loader = DropLastDataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    test_loader = DropLastDataLoader(test_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)

    backbone = build_backbone_placeholder()
    model = PerturbationModel(backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = build_optimizer(
        model,
        lr=float(cfg["model"]["lr"]),
        weight_decay=float(cfg["model"]["weight_decay"]),
    )
    
    loss_weights = cfg["training"].get("loss_weights", {
        "expr": 1.0,
        "direction": 0.5,
        "go": 0.2,
        "moa_broad": 0.3,
    })

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        device=device,
        epochs=int(cfg["training"]["epochs"]),
        loss_weights=loss_weights,
    )
    metrics = evaluate_deltas_vs_cellmean(model, test_loader, device)
    print("Final metrics:", metrics)


if __name__ == "__main__":
    main()
