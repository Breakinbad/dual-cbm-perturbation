from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
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
from perturbation.models.wrappers import (
    PerturbationModel,
    build_optimizer,
    freeze_all_parameters,
    unfreeze_modules_by_substring,
)
from perturbation.training.evaluate import evaluate_deltas_vs_cellmean
from perturbation.training.train import fit
from perturbation.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def maybe_add_project_root_to_syspath(scgpt_root: str | None) -> None:
    if not scgpt_root:
        return
    scgpt_path = Path(scgpt_root).resolve()
    project_root = scgpt_path.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def build_label_mapping(*dfs: pd.DataFrame, column: str | None) -> dict[str, int] | None:
    if not column:
        return None
    values: list[str] = []
    for df in dfs:
        if column in df.columns:
            values.extend([str(v) for v in df[column].dropna().unique().tolist()])
    if not values:
        return None
    return {label: idx for idx, label in enumerate(sorted(set(values)))}


def build_backbone_from_scgpt(cfg: dict, n_go: int, n_moa: int):
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    maybe_add_project_root_to_syspath(data_cfg.get("scgpt_root"))
    from scgpt.model.generation_model import TransformerGenerator

    vocab = load_torch_object(data_cfg["vocab_path"])
    model = TransformerGenerator(
        ntoken=int(model_cfg["ntoken"]),
        d_model=int(model_cfg["embsize"]),
        nhead=int(model_cfg["nhead"]),
        d_hid=int(model_cfg["d_hid"]),
        nlayers=int(model_cfg["nlayers"]),
        nlayers_cls=int(model_cfg["nlayers_cls"]),
        n_cls=int(model_cfg.get("n_cls", 1)),
        vocab=vocab,
        dropout=float(model_cfg["dropout"]),
        pad_token=str(model_cfg["pad_token"]),
        pad_value=int(model_cfg["pad_value"]),
        pert_pad_id=int(model_cfg["pert_pad_id"]),
        do_mvc=bool(model_cfg.get("do_mvc", False)),
        domain_spec_batchnorm=model_cfg.get("domain_spec_batchnorm", False),
        cell_emb_style=str(model_cfg.get("cell_emb_style", "cls")),
        mvc_decoder_style=str(model_cfg.get("mvc_decoder_style", "inner product")),
        ecs_threshold=float(model_cfg.get("ecs_threshold", 0.3)),
        explicit_zero_prob=bool(model_cfg.get("explicit_zero_prob", False)),
        use_fast_transformer=bool(model_cfg.get("use_fast_transformer", False)),
        fast_transformer_backend=str(model_cfg.get("fast_transformer_backend", "flash")),
        pre_norm=bool(model_cfg.get("pre_norm", False)),
    )

    if n_go > 0 and hasattr(model, "decoder_go"):
        model.go_dim = n_go
        hidden = model.d_model // 2
        dropout_p = float(model_cfg["dropout"])
        model.decoder_go = torch.nn.Sequential(
            torch.nn.Linear(model.d_model, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_p),
            torch.nn.Linear(hidden, n_go),
        )

    if n_moa > 0 and hasattr(model, "decoder_moa_broad"):
        model.moa_dim = n_moa
        hidden = model.d_model // 2
        dropout_p = float(model_cfg["dropout"])
        model.decoder_moa_broad = torch.nn.Sequential(
            torch.nn.Linear(model.d_model, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_p),
            torch.nn.Linear(hidden, n_moa),
        )

    if hasattr(model, "go_dim"):
        model.go_dim = int(model_cfg.get("go_num_classes", n_go or model.go_dim))
    if hasattr(model, "moa_dim"):
        model.moa_dim = int(model_cfg.get("moa_num_classes", n_moa or model.moa_dim))

    checkpoint_path = data_cfg.get("checkpoint_path")
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Loaded checkpoint:", checkpoint_path)
        if missing:
            print("Missing keys:", missing[:20])
        if unexpected:
            print("Unexpected keys:", unexpected[:20])

    return model


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config).raw
    seed_everything(int(cfg.get("seed", 42)))

    adata = load_anndata(cfg["data"]["adata_path"])
    smiles_dict = load_torch_object(cfg["data"]["smiles_embeddings_path"])
    averaged_df_train = load_pickle(cfg["data"]["averaged_df_train_path"])
    averaged_df_test = load_pickle(cfg["data"]["averaged_df_test_path"])

    go_label_col = cfg["data"].get("go_label_col")
    moa_label_col = cfg["data"].get("moa_label_col")
    go_to_id = build_label_mapping(averaged_df_train, averaged_df_test, column=go_label_col)
    moa_to_id = build_label_mapping(averaged_df_train, averaged_df_test, column=moa_label_col)

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
        go_to_id=go_to_id,
        moa_to_id=moa_to_id,
    )

    train_ds = GeneExpressionDataset(
        averaged_df_train,
        split="train",
        context=context,
        threshold_magnitude=float(cfg["preprocessing"]["threshold_magnitude"]),
        go_label_col=go_label_col,
        moa_label_col=moa_label_col,
    )
    test_ds = GeneExpressionDataset(
        averaged_df_test,
        split="test",
        context=context,
        threshold_magnitude=float(cfg["preprocessing"]["threshold_magnitude"]),
        go_label_col=go_label_col,
        moa_label_col=moa_label_col,
    )

    train_loader = DropLastDataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    test_loader = DropLastDataLoader(test_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)

    backbone = build_backbone_from_scgpt(cfg, n_go=len(go_to_id or {}), n_moa=len(moa_to_id or {}))
    model = PerturbationModel(
        backbone,
        expr_task_id=int(cfg["model"].get("expr_task_id", 0)),
        direction_task_id=int(cfg["model"].get("direction_task_id", 1)),
    )

    if bool(cfg["model"].get("freeze_backbone", True)):
        freeze_all_parameters(model)
        trainable = unfreeze_modules_by_substring(
            model,
            list(cfg["model"].get("trainable_substrings", ["adapter", "cond_proj", "decoder", "decoder_go", "decoder_moa_broad", "task_table", "concept_bottleneck", "z_to_drug", "z_gate", "mol_proj", "ta_decoder"])),
        )
        print(f"Unfroze {len(trainable)} parameter tensors")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = build_optimizer(
        model,
        lr=float(cfg["model"]["lr"]),
        weight_decay=float(cfg["model"]["weight_decay"]),
    )

    loss_weights = cfg["training"].get("loss_weights", {"expr": 1.0, "direction": 1.0, "go": 1.0, "moa_broad": 1.0})
    history = fit(model, train_loader, test_loader, optimizer, device, epochs=int(cfg["training"]["epochs"]), loss_weights=loss_weights)
    metrics = evaluate_deltas_vs_cellmean(model, test_loader, device)
    print("GO mapping:", go_to_id)
    print("MOA mapping:", moa_to_id)
    print("History:", history)
    print("Final delta metrics:", metrics)


if __name__ == "__main__":
    main()
