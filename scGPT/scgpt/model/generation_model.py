from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from tqdm import trange

from ..utils import map_raw_id_to_vocab_id
from .config import TASK_NAME_TO_ID, TASK_ID_TO_NAME, NUM_TASKS
from .model import (
    ContinuousValueEncoder,
    ExprDecoder,
    MVCDecoder,
    TorchTransformerEncoderLayerWithAdapter,
)
from .scdca import (
    SharedHyperFiLM,
    ConditionBuilder,
    DrugConditionalAdapterV2,
    CustomTransformerEncoder,
)


class Similarity(nn.Module):
    def __init__(self, temp: float) -> None:
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.cos(x, y) / self.temp


class GeneEncoder(nn.Module):
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


class ClsDecoder(nn.Module):
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


class TransformerGenerator(nn.Module):
    """Task-aware perturbation model.

    Main input stream:
      - gene ids
      - expression values

    Drug embedding:
      - used only as adapter conditioning
      - not added directly to token embeddings

    Outputs:
      - expression head from sequence output
      - direction head from sequence output
      - GO head from pooled transformer output
      - MOA broad head from pooled transformer output
    """

    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        nlayers_cls: int,
        n_cls: int,
        vocab: Any,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        pad_value: int = 0,
        pert_pad_id: int = 2,
        do_mvc: bool = False,
        domain_spec_batchnorm: Union[bool, str] = False,
        cell_emb_style: str = "cls",
        mvc_decoder_style: str = "inner product",
        ecs_threshold: float = 0.3,
        explicit_zero_prob: bool = False,
        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
        pre_norm: bool = False,
    ) -> None:
        super().__init__()

        if cell_emb_style not in {"cls", "avg-pool", "w-pool"}:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")

        self.model_type = "Transformer"
        self.d_model = d_model
        self.pad_token_id = vocab[pad_token]
        self.pad_value = pad_value
        self.pert_pad_id = pert_pad_id
        self.ecs_threshold = ecs_threshold
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        self.norm_scheme = "pre" if pre_norm else "post"
        self.use_fast_transformer = use_fast_transformer
        self.fast_transformer_backend = fast_transformer_backend

        # Task metadata
        self.num_tasks = NUM_TASKS
        self.task_dim = 64
        self.task_table = nn.Embedding(self.num_tasks, self.task_dim)
        self.default_expr_task_id = TASK_NAME_TO_ID.get("expr", 0)

        # Conditioning dims
        self.drug_dim = 768
        self.go_dim = 200
        self.moa_dim = 2

        # Encoders
        self.encoder = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token])
        self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        self.pert_encoder = nn.Embedding(3, d_model, padding_idx=pert_pad_id)
        self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)

        # Shared FiLM hypernetwork + adapter-aware transformer
        self.shared_hyper = SharedHyperFiLM(
            hidden_dim=d_model,
            mol_dim=self.drug_dim,
            num_tasks=self.num_tasks,
            gamma_max=0.25,
            num_layers=nlayers,
            num_positions=2,
        )

        adapter = DrugConditionalAdapterV2(
            d_model,
            128,
            self.drug_dim,
            dropout,
        )

        self.transformer_encoder = CustomTransformerEncoder(
            layer_ctor=TorchTransformerEncoderLayerWithAdapter,
            num_layers=nlayers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            adapter=adapter,
            shared_hypernetwork=self.shared_hyper,
            norm_first=True,
            batch_first=True,
        )

        # Drug embedding is used only for adapter conditioning
        self.mol_proj = nn.Linear(self.drug_dim, self.drug_dim)
        self.condition_builder = ConditionBuilder(
            drug_dim=self.drug_dim,
            task_dim=self.task_dim,
            cell_dim=0,
            out_dim=256,
        )

        # Heads
        self.decoder = ExprDecoder(d_model, explicit_zero_prob=explicit_zero_prob)

        self.decoder_direction = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3),
        )

        self.cls_decoder = ClsDecoder(d_model, n_cls, nlayers=nlayers_cls)

        self.decoder_go = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, self.go_dim),
        )

        self.decoder_moa_broad = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, self.moa_dim),
        )

        if do_mvc:
            self.mvc_decoder = MVCDecoder(
                d_model,
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob,
            )

        self.log_sigma = nn.Parameter(torch.zeros(1262))
        self.sim = Similarity(temp=0.5)
        self.creterion_cce = nn.CrossEntropyLoss()

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)

    def _normalize_padding_mask(
        self,
        src_key_padding_mask: Optional[Tensor],
        device: torch.device,
    ) -> Optional[Tensor]:
        if src_key_padding_mask is None:
            return None
        mask = src_key_padding_mask.bool()
        if mask.device != device:
            mask = mask.to(device)
        return mask

    def _build_task_ids(
        self,
        batch_size: int,
        src_device: torch.device,
        task_ids: Optional[Tensor],
    ) -> Tensor:
        if task_ids is None:
            return torch.full(
                (batch_size,),
                self.default_expr_task_id,
                dtype=torch.long,
                device=src_device,
            )
        return task_ids.to(device=src_device, dtype=torch.long)

    def _encode_inputs(self, src: Tensor, values: Tensor) -> Tensor:
        src_embed = self.encoder(src)
        self.cur_gene_token_embs = src_embed
        value_embed = self.value_encoder(values)
        total_embs = src_embed + value_embed
        return self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

    def _build_condition(
        self,
        input_pert_flags: Tensor,
        task_ids: Tensor,
        hidden_device: torch.device,
        hidden_dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor]:
        mol_emb = input_pert_flags.to(hidden_device)
        if mol_emb.dim() != 2:
            mol_emb = mol_emb.view(mol_emb.size(0), -1)

        mol_used = self.mol_proj(mol_emb)
        task_emb = self.task_table(task_ids)
        cond = self.condition_builder(drug_emb=mol_used, task_emb=task_emb)

        cond = cond.to(device=hidden_device, dtype=hidden_dtype)
        mol_used = mol_used.to(device=hidden_device, dtype=hidden_dtype)
        return mol_used, cond

    def _pool_sequence(self, transformer_output: Tensor) -> Tensor:
        if self.cell_emb_style == "cls":
            return transformer_output[:, 0, :]
        if self.cell_emb_style == "avg-pool":
            return transformer_output.mean(dim=1)
        if self.cell_emb_style == "w-pool":
            return transformer_output.mean(dim=1)
        raise ValueError(f"Unknown cell_emb_style: {self.cell_emb_style}")

    def _decode_common_outputs(self, pooled: Tensor) -> Dict[str, Tensor]:
        return {
            "go_logits": self.decoder_go(pooled),
            "moa_broad_logits": self.decoder_moa_broad(pooled),
        }

    def _decode_task_outputs(
        self,
        transformer_output: Tensor,
        task_ids: Tensor,
    ) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        unique_tasks = torch.unique(task_ids)

        for task_id in unique_tasks:
            task_id_int = int(task_id.item())
            task_name = TASK_ID_TO_NAME[task_id_int]
            task_mask = task_ids == task_id
            task_output = transformer_output[task_mask]

            if task_name == "expr":
                mlm_output = self.decoder(task_output)
                out["mlm_output"] = mlm_output["pred"]
                if "zero_probs" in mlm_output:
                    out["mlm_zero_probs"] = mlm_output["zero_probs"]

            elif task_name == "direction":
                direction_logits = self.decoder_direction(task_output)
                out["direction_logits"] = direction_logits
                out["direction_class"] = direction_logits.argmax(dim=-1)

            elif task_name in {"go", "moa_broad"}:
                # GO and MOA are decoded from pooled transformer output
                continue

            else:
                raise ValueError(f"Unknown task: {task_name}")

        return out

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        input_pert_flags: Tensor,
        src_key_padding_mask: Optional[Tensor],
        task_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size = src.size(0)
        total_embs = self._encode_inputs(src, values)
        task_ids = self._build_task_ids(batch_size, src.device, task_ids)
        mol_used, cond = self._build_condition(
            input_pert_flags=input_pert_flags,
            task_ids=task_ids,
            hidden_device=total_embs.device,
            hidden_dtype=total_embs.dtype,
        )
        padding_mask = self._normalize_padding_mask(src_key_padding_mask, total_embs.device)

        assert total_embs.dim() == 3 and total_embs.size(-1) == self.d_model, total_embs.shape
        assert mol_used.dim() == 2, mol_used.shape

        transformer_output = self.transformer_encoder(
            total_embs,
            mol_emb=mol_used,
            src_key_padding_mask=padding_mask,
            task_ids=task_ids,
            condition_embeddings=cond,
        )
        pooled = self._pool_sequence(transformer_output)
        return transformer_output, pooled, task_ids

    def forward(
        self,
        src: Tensor,
        values: Tensor,
        input_pert_flags: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        task_ids: Optional[Tensor] = None,
        **unused: Any,
    ) -> Mapping[str, Tensor]:
        transformer_output, pooled, task_ids = self._encode(
            src=src,
            values=values,
            input_pert_flags=input_pert_flags,
            src_key_padding_mask=src_key_padding_mask,
            task_ids=task_ids,
        )
        out = self._decode_common_outputs(pooled)
        out.update(self._decode_task_outputs(transformer_output, task_ids))
        return out

    def encode_batch(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_size: int,
        output_to_cpu: bool = True,
    ) -> Tensor:
        outputs = []
        num_samples = src.size(0)
        device = next(self.parameters()).device

        for i in trange(0, num_samples, batch_size):
            batch_output, _, _ = self._encode(
                src=src[i : i + batch_size].to(device),
                values=values[i : i + batch_size].to(device),
                input_pert_flags=torch.zeros(
                    src[i : i + batch_size].size(0),
                    self.drug_dim,
                    device=device,
                ),
                src_key_padding_mask=src_key_padding_mask[i : i + batch_size].to(device),
            )
            if output_to_cpu:
                batch_output = batch_output.cpu()
            outputs.append(batch_output)

        return torch.cat(outputs, dim=0)

    def pred_perturb(
        self,
        batch_data: Any,
        include_zero_gene: str = "batch-wise",
        gene_ids: Optional[Tensor] = None,
        amp: bool = True,
    ) -> Tensor:
        self.eval()
        device = next(self.parameters()).device
        batch_data.to(device)

        batch_size = len(batch_data.pert)
        x: Tensor = batch_data.x
        ori_gene_values = x[:, 0].view(batch_size, -1)
        pert_flags = x[:, 1].long().view(batch_size, -1)

        if include_zero_gene not in {"all", "batch-wise"}:
            raise ValueError(f"Unsupported include_zero_gene={include_zero_gene}")

        if gene_ids is None:
            raise ValueError("gene_ids must be provided for pred_perturb")

        if include_zero_gene == "all":
            input_gene_ids = torch.arange(ori_gene_values.size(1), device=device)
        else:
            input_gene_ids = ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]

        input_values = ori_gene_values[:, input_gene_ids]
        input_pert_flags = pert_flags[:, input_gene_ids]
        mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids).repeat(batch_size, 1)
        src_key_padding_mask = torch.zeros_like(input_values, dtype=torch.bool, device=device)

        with torch.cuda.amp.autocast(enabled=amp):
            output_dict = self(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                do_sample=True,
            )

        pred_gene_values = torch.zeros_like(ori_gene_values)
        pred_gene_values[:, input_gene_ids] = output_dict["mlm_output"].float()
        return pred_gene_values