from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Attention pooling over the sequence dimension."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_scores = self.attention(x)  # [B, T, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        return (x * attn_weights).sum(dim=1)


class CustomTransformerEncoder(nn.Module):
    """Adapter-aware transformer encoder stack.

    This wrapper exists because the standard PyTorch TransformerEncoder does not
    forward custom keyword arguments like `task_ids`, `mol_emb`, or
    `condition_embeddings` to each layer.
    """

    def __init__(
        self,
        layer_ctor: Callable[..., nn.Module],
        num_layers: int,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        adapter: Optional[nn.Module],
        shared_hypernetwork: Optional[nn.Module],
        norm_first: bool = True,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()

        for layer_id in range(num_layers):
            # IMPORTANT FIX: each layer gets its own adapter instance
            layer_adapter = copy.deepcopy(adapter) if adapter is not None else None

            layer = layer_ctor(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=batch_first,
                norm_first=norm_first,
                adapter=layer_adapter,
                shared_hypernetwork=shared_hypernetwork,
                layer_id=layer_id,
                pos_id=1,
            )

            hyper = getattr(getattr(layer, "adapter", None), "hypernetwork", None)
            if hyper is not None:
                if hasattr(hyper, "layer_id"):
                    hyper.layer_id = layer_id
                if hasattr(hyper, "pos_id"):
                    hyper.pos_id = 1

            self.layers.append(layer)

    def forward(
        self,
        src: torch.Tensor,
        *,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        mol_emb: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None,
        condition_embeddings: Optional[torch.Tensor] = None,
        cell_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = src
        for layer in self.layers:
            x = layer(
                x,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                mol_emb=mol_emb,
                task_ids=task_ids,
                condition_embeddings=condition_embeddings,
                cell_emb=cell_emb,
            )
        return x


@dataclass
class FiLMConfig:
    """Configuration for a FiLM-conditioned adapter."""

    hidden_dim: int
    mol_dim: int
    bottleneck_dim: int = 128
    ff_multiplier: int = 4
    gamma_max: float = 1.5
    alpha_init: float = 1e-3
    use_cond_ln: bool = True


class ConditionBuilder(nn.Module):
    """Project task/drug/(optional) cell context into a single condition vector."""

    def __init__(
        self,
        drug_dim: int = 768,
        task_dim: int = 32,
        cell_dim: int = 0,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        in_dim = (drug_dim if drug_dim else 0) + task_dim + (cell_dim if cell_dim else 0)
        hidden_dim = max(out_dim, in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        *,
        drug_emb: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None,
        task_emb: Optional[torch.Tensor] = None,
        cell_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pieces = []
        if drug_emb is not None:
            pieces.append(drug_emb)
        if task_emb is None:
            raise RuntimeError("Pass task_emb to ConditionBuilder.forward().")
        pieces.append(task_emb)
        if cell_emb is not None:
            pieces.append(cell_emb)
        return self.proj(torch.cat(pieces, dim=-1))


class HypernetShim(nn.Module):
    """Bind a shared hypernetwork to a specific transformer layer/position."""

    def __init__(self, shared_hnet: nn.Module, layer_id: int, pos_id: int) -> None:
        super().__init__()
        self.shared_hnet = shared_hnet
        self.layer_id = int(layer_id)
        self.pos_id = int(pos_id)

    def forward(
        self,
        task_ids: torch.LongTensor,
        mol_emb: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.shared_hnet(task_ids, self.layer_id, self.pos_id, mol_emb, hidden)


class SharedHyperFiLM(nn.Module):
    """Shared hypernetwork that produces FiLM gamma/beta parameters."""

    def __init__(
        self,
        hidden_dim: Optional[int] = None,
        mol_dim: Optional[int] = None,
        num_tasks: int = 1,
        task_emb_dim: int = 64,
        layer_emb_dim: int = 16,
        pos_emb_dim: int = 8,
        hnet_width: int = 256,
        gamma_max: float = 1.5,
        bottleneck_dim: Optional[int] = None,
        drug_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        num_positions: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if mol_dim is None:
            mol_dim = drug_dim
        if mol_dim is None:
            raise ValueError("Provide mol_dim or drug_dim.")

        film_dim = hidden_dim if hidden_dim is not None else bottleneck_dim
        if film_dim is None:
            raise ValueError("Provide hidden_dim or bottleneck_dim.")

        max_layers = num_layers if (num_layers is not None and num_layers > 0) else 256
        max_positions = num_positions if (num_positions is not None and num_positions > 0) else 16

        self.hidden_dim = int(film_dim)
        self.mol_dim = int(mol_dim)
        self.gamma_max = float(gamma_max)

        self.task_embed = nn.Embedding(num_tasks, task_emb_dim)
        self.layer_embed = nn.Embedding(max_layers, layer_emb_dim)
        self.pos_embed = nn.Embedding(max_positions, pos_emb_dim)

        in_dim = task_emb_dim + layer_emb_dim + pos_emb_dim + self.mol_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hnet_width),
            nn.GELU(),
            nn.Linear(hnet_width, 2 * self.hidden_dim),
        )

        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        task_ids: torch.LongTensor,
        layer_id: int,
        pos_id: int,
        mol_emb: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = mol_emb.size(0)
        device = mol_emb.device

        task_emb = self.task_embed(task_ids.to(device))
        safe_layer_id = max(0, min(layer_id, self.layer_embed.num_embeddings - 1))
        safe_pos_id = max(0, min(pos_id, self.pos_embed.num_embeddings - 1))

        layer_emb = self.layer_embed(
            torch.full((batch_size,), safe_layer_id, device=device, dtype=torch.long)
        )
        pos_emb = self.pos_embed(
            torch.full((batch_size,), safe_pos_id, device=device, dtype=torch.long)
        )

        hnet_input = torch.cat([task_emb, layer_emb, pos_emb, mol_emb], dim=-1)
        film = self.mlp(hnet_input)
        gamma, beta = film.chunk(2, dim=-1)
        gamma = torch.tanh(gamma) * self.gamma_max
        return gamma, beta


class DrugConditionalAdapter(nn.Module):
    """Residual adapter conditioned by FiLM parameters."""

    def __init__(
        self,
        hidden_dim: int,
        mol_dim: int,
        bottleneck_dim: int = 128,
        gamma_max: float = 1.5,
        alpha_init: float = 1e-3,
        hypernetwork: Optional[nn.Module] = None,
        use_cond_ln: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mol_dim = mol_dim
        self.bottleneck_dim = bottleneck_dim
        self.gamma_max = float(gamma_max)
        self.alpha = nn.Parameter(torch.full((), float(alpha_init)))

        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.norm = nn.LayerNorm(bottleneck_dim) if use_cond_ln else nn.Identity()

        self.hypernetwork = hypernetwork
        if hypernetwork is None:
            self.local_film = nn.Linear(mol_dim, hidden_dim * 2)
            nn.init.zeros_(self.local_film.weight)
            nn.init.zeros_(self.local_film.bias)
        else:
            self.local_film = None

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def _compute_film(
        self,
        task_ids: torch.LongTensor,
        mol_emb: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.hypernetwork is not None:
            gamma, beta = self.hypernetwork(task_ids, mol_emb, hidden)
        else:
            film = self.local_film(mol_emb)
            gamma, beta = film.chunk(2, dim=-1)
            gamma = torch.tanh(gamma) * self.gamma_max
        return gamma.unsqueeze(1), beta.unsqueeze(1)

    def forward(
        self,
        hidden: torch.Tensor,
        task_ids: torch.LongTensor,
        mol_emb: Optional[torch.Tensor] = None,
        molecule_embeddings: Optional[torch.Tensor] = None,
        condition_embeddings: Optional[torch.Tensor] = None,
        cell_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        adapter_emb = (
            condition_embeddings
            if condition_embeddings is not None
            else molecule_embeddings
            if molecule_embeddings is not None
            else mol_emb
        )
        if adapter_emb is None:
            raise ValueError("Provide condition_embeddings, molecule_embeddings, or mol_emb.")

        # Use raw mol_emb for FiLM hypernetwork when available.
        hyper_emb = mol_emb if mol_emb is not None else adapter_emb

        residual = hidden
        out = self.down(hidden)
        out = F.gelu(out)
        out = self.norm(out)
        out = self.up(out)

        gamma, beta = self._compute_film(task_ids, hyper_emb, hidden=residual)
        out = (1.0 + gamma) * out + beta
        return residual + self.alpha * out


class DrugConditionalAdapterV2(DrugConditionalAdapter):
    """Backward-compatible alias for the current adapter implementation."""

    pass


def build_adapter_from_config(
    cfg: FiLMConfig,
    shared_hnet: Optional[SharedHyperFiLM] = None,
) -> DrugConditionalAdapter:
    """Build a DrugConditionalAdapter from a FiLMConfig."""
    hypernetwork = None
    return DrugConditionalAdapter(
        hidden_dim=cfg.hidden_dim,
        mol_dim=cfg.mol_dim,
        bottleneck_dim=cfg.bottleneck_dim,
        gamma_max=cfg.gamma_max,
        alpha_init=cfg.alpha_init,
        hypernetwork=hypernetwork,
        use_cond_ln=cfg.use_cond_ln,
    )