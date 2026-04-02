from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .dsbn import DomainSpecificBatchNorm1d
from .grad_reverse import grad_reverse
from .scdca import CustomTransformerEncoder, DrugConditionalAdapterV2 as DrugConditionalAdapter
from .scdca import HypernetShim, SharedHyperFiLM


class TorchTransformerEncoderLayerWithAdapter(nn.TransformerEncoderLayer):
    """PyTorch TransformerEncoderLayer with an optional conditional adapter."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_first: bool = True,
        norm_first: bool = True,
        adapter: Optional[DrugConditionalAdapter] = None,
        shared_hypernetwork: Optional[SharedHyperFiLM] = None,
        layer_id: int = 0,
        pos_id: int = 1,
    ) -> None:
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self.d_model = d_model
        self.adapter = adapter

        if self.adapter is not None and shared_hypernetwork is not None:
            self.adapter.hypernetwork = HypernetShim(
                shared_hypernetwork,
                layer_id=layer_id,
                pos_id=pos_id,
            )

        self._cond_task_ids: Optional[torch.LongTensor] = None
        self._cond_mol_emb: Optional[torch.Tensor] = None
        self._cond_generic: Optional[torch.Tensor] = None

    @torch.no_grad()
    def set_condition(
        self,
        task_ids: Optional[torch.LongTensor],
        mol_emb: Optional[torch.Tensor] = None,
        condition_embeddings: Optional[torch.Tensor] = None,
    ) -> None:
        self._cond_task_ids = task_ids
        self._cond_mol_emb = mol_emb
        self._cond_generic = condition_embeddings

    @torch.no_grad()
    def clear_condition(self) -> None:
        self._cond_task_ids = None
        self._cond_mol_emb = None
        self._cond_generic = None

    def _normalize_src(self, src: torch.Tensor) -> torch.Tensor:
        if src.dim() != 3:
            raise ValueError(f"Expected 3D src [B, T, D], got {tuple(src.shape)}")

        batch_size, seq_len, last_dim = src.shape
        if last_dim == self.d_model:
            return src

        if seq_len == self.d_model and batch_size > 0:
            return src.transpose(1, 2).contiguous()

        raise ValueError(
            f"Expected src last dim == d_model={self.d_model}, got {last_dim}. "
            f"src shape={tuple(src.shape)}"
        )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        *,
        mol_emb: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None,
        condition_embeddings: Optional[torch.Tensor] = None,
        cell_emb: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if src_key_padding_mask is not None and src_key_padding_mask.dtype is not torch.bool:
            src_key_padding_mask = src_key_padding_mask.bool()

        src = self._normalize_src(src)
        out = super().forward(
            src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        task_ids = task_ids if task_ids is not None else self._cond_task_ids
        cond_generic = condition_embeddings if condition_embeddings is not None else self._cond_generic
        mol_emb = mol_emb if mol_emb is not None else self._cond_mol_emb

        if self.adapter is not None and task_ids is not None:
            task_ids = task_ids.to(out.device, dtype=torch.long)

            if cond_generic is not None:
                out = self.adapter(
                    out,
                    condition_embeddings=cond_generic.to(out.device, dtype=out.dtype),
                    task_ids=task_ids,
                    mol_emb=mol_emb.to(out.device, dtype=out.dtype) if mol_emb is not None else None,
                    cell_emb=cell_emb.to(out.device, dtype=out.dtype) if cell_emb is not None else None,
                )
            elif mol_emb is not None:
                out = self.adapter(
                    out,
                    task_ids=task_ids,
                    mol_emb=mol_emb.to(out.device, dtype=out.dtype),
                    molecule_embeddings=mol_emb.to(out.device, dtype=out.dtype),
                    cell_emb=cell_emb.to(out.device, dtype=out.dtype) if cell_emb is not None else None,
                )

        self.clear_condition()
        return out


class FlashTransformerEncoderLayer(nn.Module):
    """Compatibility-friendly encoder layer with adapter conditioning."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        adapter: Optional[DrugConditionalAdapter] = None,
        shared_hypernetwork: Optional[SharedHyperFiLM] = None,
        layer_id: int = 0,
        pos_id: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

        self.adapter = adapter
        if self.adapter is not None and shared_hypernetwork is not None:
            self.adapter.hypernetwork = HypernetShim(
                shared_hypernetwork,
                layer_id=layer_id,
                pos_id=pos_id,
            )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        *,
        task_ids: Optional[torch.LongTensor] = None,
        mol_emb: Optional[torch.Tensor] = None,
        condition_embeddings: Optional[torch.Tensor] = None,
        cell_emb: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if src_key_padding_mask is not None and src_key_padding_mask.dtype is not torch.bool:
            src_key_padding_mask = src_key_padding_mask.bool()

        residual = src
        src2, _ = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        src = self.norm1(residual + self.dropout1(src2))

        residual = src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(residual + self.dropout2(src2))

        if self.adapter is not None and task_ids is not None:
            if condition_embeddings is not None:
                src = self.adapter(
                    src,
                    task_ids=task_ids,
                    condition_embeddings=condition_embeddings,
                    mol_emb=mol_emb,
                    cell_emb=cell_emb,
                )
            elif mol_emb is not None:
                src = self.adapter(
                    src,
                    task_ids=task_ids,
                    mol_emb=mol_emb,
                    molecule_embeddings=mol_emb,
                    cell_emb=cell_emb,
                )
        return src


class TransformerModel(nn.Module):
    """Adapter-aware transformer backbone built on the CustomTransformerEncoder."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        *,
        adapter_factory=None,
        shared_hypernetwork: Optional[SharedHyperFiLM] = None,
        use_fast_transformer: bool = False,
    ) -> None:
        super().__init__()

        layer_ctor = FlashTransformerEncoderLayer if use_fast_transformer else TorchTransformerEncoderLayerWithAdapter

        # IMPORTANT: CustomTransformerEncoder in scdca.py expects a template adapter,
        # and then deep-copies it per layer internally.
        base_adapter = adapter_factory(0) if adapter_factory is not None else None

        self.encoder = CustomTransformerEncoder(
            layer_ctor=layer_ctor,
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            adapter=base_adapter,
            shared_hypernetwork=shared_hypernetwork,
            norm_first=True,
            batch_first=True,
        )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        *,
        task_ids: Optional[torch.LongTensor] = None,
        mol_emb: Optional[torch.Tensor] = None,
        condition_embeddings: Optional[torch.Tensor] = None,
        cell_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if src_key_padding_mask is not None and src_key_padding_mask.dtype != torch.bool:
            src_key_padding_mask = src_key_padding_mask.bool()

        return self.encoder(
            src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            task_ids=task_ids,
            mol_emb=mol_emb,
            condition_embeddings=condition_embeddings,
            cell_emb=cell_emb,
        )

class FlashTransformerEncoderLayer(nn.Module):
    """Encoder layer with the same call signature as the adapter-aware torch layer.

    This is a compatibility-friendly implementation that supports the same
    conditioning interface when used inside the custom encoder wrapper.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        adapter: Optional[DrugConditionalAdapter] = None,
        shared_hypernetwork: Optional[SharedHyperFiLM] = None,
        layer_id: int = 0,
        pos_id: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

        self.adapter = adapter
        if self.adapter is not None and shared_hypernetwork is not None:
            self.adapter.hypernetwork = HypernetShim(
                shared_hypernetwork,
                layer_id=layer_id,
                pos_id=pos_id,
            )

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        *,
        task_ids: Optional[torch.LongTensor] = None,
        mol_emb: Optional[torch.Tensor] = None,
        condition_embeddings: Optional[torch.Tensor] = None,
        cell_emb: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = src
        src2, _ = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        src = self.norm1(residual + self.dropout1(src2))

        residual = src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(residual + self.dropout2(src2))

        if self.adapter is not None and task_ids is not None:
            if condition_embeddings is not None:
                src = self.adapter(
                    src,
                    task_ids=task_ids,
                    condition_embeddings=condition_embeddings,
                    mol_emb=mol_emb,
                    cell_emb=cell_emb,
                )
            elif mol_emb is not None:
                src = self.adapter(
                    src,
                    task_ids=task_ids,
                    mol_emb=mol_emb,
                    cell_emb=cell_emb,
                )
        return src


class TransformerModel(nn.Module):
    """Adapter-aware transformer backbone built on a custom encoder wrapper."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        *,
        adapter_factory,
        shared_hypernetwork: Optional[SharedHyperFiLM] = None,
        use_fast_transformer: bool = True,
    ) -> None:
        super().__init__()

        layers = []
        layer_cls = FlashTransformerEncoderLayer if use_fast_transformer else TorchTransformerEncoderLayerWithAdapter
        for layer_id in range(num_layers):
            adapter = adapter_factory(layer_id) if adapter_factory is not None else None
            layer = layer_cls(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                adapter=adapter,
                shared_hypernetwork=shared_hypernetwork,
                layer_id=layer_id,
                pos_id=1,
            )
            layers.append(layer)

        self.encoder = CustomTransformerEncoder(layers)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        *,
        task_ids: Optional[torch.LongTensor] = None,
        mol_emb: Optional[torch.Tensor] = None,
        condition_embeddings: Optional[torch.Tensor] = None,
        cell_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if src_key_padding_mask is not None and src_key_padding_mask.dtype != torch.bool:
            src_key_padding_mask = src_key_padding_mask.bool()

        return self.encoder(
            src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            task_ids=task_ids,
            mol_emb=mol_emb,
            condition_embeddings=condition_embeddings,
            cell_emb=cell_emb,
        )


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generate an upper-triangular causal mask."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class FastTransformerEncoderWrapper(nn.Module):
    """Wrapper around fast-transformers linear attention encoder."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.fast_transformer_encoder = self.build_fast_transformer_encoder(
            d_model=d_model,
            nhead=nhead,
            d_hid=d_hid,
            nlayers=nlayers,
            dropout=dropout,
        )

    @staticmethod
    def build_fast_transformer_encoder(
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float,
    ) -> nn.Module:
        from fast_transformers.builders import TransformerEncoderBuilder

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model must be divisible by nhead, got d_model={d_model}, nhead={nhead}"
            )

        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=nlayers,
            n_heads=nhead,
            query_dimensions=d_model // nhead,
            value_dimensions=d_model // nhead,
            feed_forward_dimensions=d_hid,
            attention_type="linear",
            attention_dropout=dropout,
            dropout=dropout,
            activation="gelu",
        )
        assert builder.attention_type == "linear"
        return builder.get()

    @staticmethod
    def build_length_mask(
        src: Tensor,
        src_key_padding_mask: torch.BoolTensor,
    ) -> "LengthMask":
        from fast_transformers.masking import LengthMask

        seq_len = src.shape[1]
        num_paddings = src_key_padding_mask.sum(dim=1)
        actual_seq_len = seq_len - num_paddings
        length_mask = LengthMask(actual_seq_len, max_len=seq_len, device=src.device)

        if src_key_padding_mask[length_mask.bool_matrix].sum() != 0:
            raise ValueError(
                "Found padding tokens in the middle of the sequence. "
                "src_key_padding_mask and length_mask are not compatible."
            )
        return length_mask

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: torch.BoolTensor,
    ) -> Tensor:
        if src_key_padding_mask.shape != src.shape[:2]:
            raise ValueError(
                f"src_key_padding_mask shape {src_key_padding_mask.shape} does not match "
                f"src shape {src.shape[:2]}"
            )
        if src_key_padding_mask.dtype != torch.bool:
            raise ValueError(
                f"src_key_padding_mask must be torch.bool, got {src_key_padding_mask.dtype}"
            )

        length_mask = self.build_length_mask(src, src_key_padding_mask)
        return self.fast_transformer_encoder(src, length_mask=length_mask)


class LoRALayer(nn.Module):
    """Very small LoRA-style residual layer with an external condition vector."""

    def __init__(self, input_dim: int, output_dim: int, rank: int = 4) -> None:
        super().__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.randn(input_dim, rank))
        self.B = nn.Parameter(torch.randn(rank, output_dim))

    def forward(self, x: torch.Tensor, molecular_embedding: torch.Tensor) -> torch.Tensor:
        return x + x @ self.A @ self.B + molecular_embedding


class Hypernetwork(nn.Module):
    """Single hidden projection used by the legacy adapter implementation."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc1(x))


class ResidualBlock(nn.Module):
    """Simple residual MLP block."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.non_linearity = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.non_linearity(out)
        out = self.fc2(out)
        out = self.norm2(out)
        return out + residual


class Adapter(nn.Module):
    """Legacy adapter conditioned by a hypernetwork-generated bias."""

    def __init__(self, mol_dim: int, input_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        self.resnet = ResidualBlock(bottleneck_dim, bottleneck_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, bottleneck_dim)
        self.linear2 = nn.Linear(bottleneck_dim, input_dim)
        self.non_linearity = nn.ReLU()
        self.hypernetwork = Hypernetwork(input_dim, bottleneck_dim + input_dim)

    def forward(self, x: torch.Tensor, molecule_embeddings: torch.Tensor) -> torch.Tensor:
        hyper_biases = self.hypernetwork(molecule_embeddings)
        down_proj_bias = hyper_biases[:, :, : self.bottleneck_dim]
        up_proj_bias = hyper_biases[:, :, : self.input_dim]

        x = self.norm1(x)
        x = self.linear1(x) + down_proj_bias
        x = self.resnet(x)
        x = self.linear2(x) + up_proj_bias
        return x


class GeneEncoder(nn.Module):
    """Embedding layer for gene token ids."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.enc_norm(self.embedding(x))


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.pe[: x.size(0)])


class ContinuousValueEncoder(nn.Module):
    """Encode real-valued expression magnitudes into embeddings."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class CategoryValueEncoder(nn.Module):
    """Categorical value encoder."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.enc_norm(self.embedding(x.long()))


class BatchLabelEncoder(nn.Module):
    """Batch-label embedding encoder."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.enc_norm(self.embedding(x))


class Similarity(nn.Module):
    """Temperature-scaled cosine similarity."""

    def __init__(self, temp: float) -> None:
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.cos(x, y) / self.temp


class ExprDecoder(nn.Module):
    """Expression decoder head."""

    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ) -> None:
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        pred_value = self.fc(x).squeeze(-1)

        if not self.explicit_zero_prob:
            return {"pred": pred_value}

        zero_logits = self.zero_logit(x).squeeze(-1)
        zero_probs = torch.sigmoid(zero_logits)
        return {"pred": pred_value, "zero_probs": zero_probs}


class ClsDecoder(nn.Module):
    """MLP classifier head."""

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ) -> None:
        super().__init__()
        self._decoder = nn.ModuleList()
        for _ in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class MVCDecoder(nn.Module):
    """Masked value prediction decoder for cell embeddings."""

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ) -> None:
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model

        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(
        self,
        cell_emb: Tensor,
        gene_embs: Tensor,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs

        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)

            if not self.explicit_zero_prob:
                return {"pred": pred_value}

            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return {"pred": pred_value, "zero_probs": zero_probs}

        if self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)
            hidden = self.hidden_activation(self.fc1(torch.cat([cell_emb, query_vecs], dim=2)))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(hidden).squeeze(2)

        if self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)
            hidden = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(hidden).squeeze(2)

        raise ValueError(f"Unknown arch_style: {self.arch_style}")


class AdversarialDiscriminator(nn.Module):
    """Discriminator for adversarial batch correction."""

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = False,
    ) -> None:
        super().__init__()
        self._decoder = nn.ModuleList()
        for _ in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor) -> Tensor:
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)
