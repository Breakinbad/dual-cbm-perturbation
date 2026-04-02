from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
from scipy import sparse
from tqdm import tqdm


ArrayLike = np.ndarray | torch.Tensor


def _to_dense(x: ArrayLike) -> np.ndarray:
    if sparse.issparse(x):
        x = x.toarray()
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).squeeze()


def compute_top_de_genes(
    adata,
    cell_type_key: str = "cell_type",
    condition_key: str = "condition",
    control_label: str = "control",
    top_k: int = 50,
) -> dict[tuple[str, str], list[str]]:
    """Compute a simple top-K DEG set per (condition, cell_type).

    This is a cleaned replacement for the DEG-building cells in the notebook.
    You can swap this out with a more sophisticated DE method later.
    """
    de_genes: dict[tuple[str, str], list[str]] = {}
    X = adata.X
    obs = adata.obs
    gene_names = np.asarray(adata.var_names)

    for cell_type in tqdm(obs[cell_type_key].unique(), desc="cell types"):
        idx_cell = (obs[cell_type_key] == cell_type).values
        idx_ctrl = ((obs[cell_type_key] == cell_type) & (obs[condition_key] == control_label)).values
        if idx_ctrl.sum() == 0:
            continue

        ctrl_mean = _to_dense(X[idx_ctrl].mean(axis=0))
        for condition in obs.loc[idx_cell, condition_key].unique():
            if condition == control_label:
                continue
            idx_cond = ((obs[cell_type_key] == cell_type) & (obs[condition_key] == condition)).values
            if idx_cond.sum() == 0:
                continue
            cond_mean = _to_dense(X[idx_cond].mean(axis=0))
            delta = cond_mean - ctrl_mean
            top_idx = np.argsort(np.abs(delta))[-top_k:]
            de_genes[(condition, cell_type)] = gene_names[top_idx].tolist()
    return de_genes


def build_gene_index_maps(adata, de_genes: dict[tuple[str, str], list[str]]):
    gene_name_to_id = {gene_name: i for i, gene_name in enumerate(adata.var_names)}
    union_gene_names = sorted({gene for genes in de_genes.values() for gene in genes})

    molecule_to_genes = {
        key: [gene_name_to_id[g] for g in union_gene_names if g in gene_name_to_id]
        for key in de_genes
    }
    molecule_to_genes_de = {
        key: [gene_name_to_id[g] for g in genes if g in gene_name_to_id]
        for key, genes in de_genes.items()
    }
    return gene_name_to_id, union_gene_names, molecule_to_genes, molecule_to_genes_de


def compute_control_statistics(
    adata,
    cell_type_key: str = "cell_type",
    condition_key: str = "condition",
    control_label: str = "control",
):
    control_mean: dict[str, torch.Tensor] = {}
    control_std: dict[str, torch.Tensor] = {}
    obs = adata.obs

    for cell_type in tqdm(obs[cell_type_key].unique(), desc="controls"):
        idx = ((obs[cell_type_key] == cell_type) & (obs[condition_key] == control_label)).values
        if idx.sum() == 0:
            continue
        x = adata.X[idx]
        x_dense = _to_dense(x)
        if x_dense.ndim == 1:
            x_dense = x_dense[None, :]
        control_mean[cell_type] = torch.tensor(x_dense.mean(axis=0), dtype=torch.float32)
        control_std[cell_type] = torch.tensor(x_dense.std(axis=0), dtype=torch.float32)
    return control_mean, control_std


def compute_average_expression(adata, cell_type_key: str = "cell_type", condition_key: str = "condition"):
    averages: dict[tuple[str, str], torch.Tensor] = {}
    obs = adata.obs
    for cell_type in obs[cell_type_key].unique():
        for condition in obs[condition_key].unique():
            idx = ((obs[cell_type_key] == cell_type) & (obs[condition_key] == condition)).values
            if idx.sum() == 0:
                continue
            expr = adata.X[idx].mean(axis=0)
            averages[(cell_type, condition)] = torch.tensor(_to_dense(expr), dtype=torch.float32)
    return averages


def build_cell_type_lookup(cell_types: Iterable[str]) -> dict[str, int]:
    return {cell_type: idx for idx, cell_type in enumerate(sorted(cell_types))}
