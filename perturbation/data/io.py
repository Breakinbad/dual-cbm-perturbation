from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import scanpy as sc
import torch


def load_anndata(path: str | Path):
    return sc.read(str(path), cache=True)


def load_torch_object(path: str | Path) -> Any:
    return torch.load(path)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
