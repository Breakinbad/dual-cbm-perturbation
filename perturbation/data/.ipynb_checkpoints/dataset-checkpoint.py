from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class DropLastDataLoader(DataLoader):
    """Notebook-compatible loader that discards the final undersized batch."""

    def __iter__(self):
        for batch in super().__iter__():
            try:
                batch_size = next(iter(batch.values())).size(0) if isinstance(batch, dict) else batch[0].size(0)
                if batch_size >= self.batch_size:
                    yield batch
            except Exception:
                yield batch


@dataclass
class DatasetContext:
    molecule_to_genes: dict[tuple[str, str], list[int]]
    average_expression: dict[tuple[str, str], torch.Tensor]
    control_data_dict: dict[str, torch.Tensor]
    control_std_dict: dict[str, torch.Tensor]
    smiles_dict: dict[str, Any]
    average_expression_train: dict[str, torch.Tensor]
    cell_dic: dict[str, int]
    cell_types: list[str]


class GeneExpressionDataset(Dataset):
    def __init__(
        self,
        averaged_df: pd.DataFrame,
        split: str,
        context: DatasetContext,
        threshold_magnitude: float = 0.05,
        task3_type: str = "cell_type",
    ):
        self.averaged_df = averaged_df.reset_index(drop=True)
        self.split = split
        self.context = context
        self.threshold_magnitude = threshold_magnitude
        self.task3_type = task3_type

    def __len__(self) -> int:
        return len(self.averaged_df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.averaged_df.iloc[idx]
        cell_type = row["cell_type"]
        molecule = row["condition"]
        smiles = row["SMILES"]

        genes_id = self.context.molecule_to_genes[(molecule, cell_type)]
        perturbed_expression = self.context.average_expression[(cell_type, molecule)][genes_id]
        control_expression = self.context.control_data_dict[cell_type][genes_id]
        control_std = self.context.control_std_dict[cell_type][genes_id]
        avg_train = self.context.average_expression_train[cell_type][genes_id]

        smiles_embedding = torch.as_tensor(self.context.smiles_dict[smiles], dtype=torch.float32)
        delta = perturbed_expression - control_expression

        direction_classes = np.ones_like(delta, dtype=np.int64)
        direction_classes[delta > self.threshold_magnitude] = 2
        direction_classes[delta < -self.threshold_magnitude] = 0

        cell_id = self.context.cell_dic[cell_type]
        if self.task3_type == "cell_type":
            target_task3 = torch.tensor(cell_id, dtype=torch.long)
        elif self.task3_type == "magnitude":
            avg_abs_delta = np.abs(delta).mean()
            if avg_abs_delta < 0.05:
                target_task3 = torch.tensor(0, dtype=torch.long)
            elif avg_abs_delta < 0.15:
                target_task3 = torch.tensor(1, dtype=torch.long)
            else:
                target_task3 = torch.tensor(2, dtype=torch.long)
        else:
            raise ValueError(f"Unsupported task3_type: {self.task3_type}")

        return {
            "gene_ids": torch.tensor(genes_id, dtype=torch.long),
            "values": torch.as_tensor(control_expression, dtype=torch.float32),
            "drug_emb": smiles_embedding,
            "target_expr": torch.as_tensor(perturbed_expression, dtype=torch.float32),
            "target_direction": torch.tensor(direction_classes, dtype=torch.long),
            "target_cell_type": target_task3,
            "control_std": torch.as_tensor(control_std, dtype=torch.float32),
            "padding_mask": torch.zeros(len(genes_id), dtype=torch.bool),
            "avg_exp_train": torch.as_tensor(avg_train, dtype=torch.float32),
        }
