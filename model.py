"""Transformer encoders for CLIP-style PPG/GSR representation learning."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class SignalTransformerEncoder(nn.Module):
    """Transformer encoder for one-dimensional physiological signals.

    Input shape can be either [batch, length] or [batch, length, channels].
    The output is an L2-normalized embedding with shape [batch, embed_dim].
    """

    def __init__(
        self,
        input_channels: int = 1,
        embed_dim: int = 128,
        transformer_dim: int = 128,
        patch_size: int = 600,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")

        self.patch_size = patch_size
        self.patch_embed = nn.Conv1d(
            in_channels=input_channels,
            out_channels=transformer_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.pos_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=int(transformer_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(transformer_dim)
        self.projection = nn.Sequential(nn.Linear(
            transformer_dim, transformer_dim), 
            nn.GELU(), 
            nn.Linear(transformer_dim, embed_dim))  
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    @staticmethod
    def _sinusoidal_positional_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
        position = torch.arange(length, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
        pe = torch.zeros(length, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.ndim != 3:
            raise ValueError("Input must have shape [batch, length] or [batch, length, channels]")

        # Conv1d expects [batch, channels, length].
        x = x.transpose(1, 2)
        x = self.patch_embed(x)
        x = x.transpose(1, 2)

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self._sinusoidal_positional_encoding(x.size(1), x.size(2), x.device)
        x = self.pos_dropout(x)

        x = self.encoder(x)
        cls_out = self.norm(x[:, 0])
        z = self.projection(cls_out)
        return F.normalize(z, dim=-1)


class PPGGSRCLIP(nn.Module):
    """Two-tower CLIP-style model with one PPG encoder and one GSR encoder."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        encoder_kwargs = dict(
            input_channels=cfg.input_channels,
            embed_dim=cfg.embed_dim,
            transformer_dim=cfg.transformer_dim,
            patch_size=cfg.patch_size,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
        )
        self.ppg_encoder = SignalTransformerEncoder(**encoder_kwargs)
        self.gsr_encoder = SignalTransformerEncoder(**encoder_kwargs)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / cfg.temperature)))

    def encode_ppg(self, ppg: torch.Tensor) -> torch.Tensor:
        return self.ppg_encoder(ppg)

    def encode_gsr(self, gsr: torch.Tensor) -> torch.Tensor:
        return self.gsr_encoder(gsr)

    def forward(self, ppg: torch.Tensor, gsr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ppg_z = self.encode_ppg(ppg)
        gsr_z = self.encode_gsr(gsr)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * ppg_z @ gsr_z.t()
        return logits, ppg_z, gsr_z


def clip_contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE loss for matched PPG/GSR batches."""

    labels = torch.arange(logits.size(0), device=logits.device)
    loss_ppg_to_gsr = F.cross_entropy(logits, labels)
    loss_gsr_to_ppg = F.cross_entropy(logits.t(), labels)
    return (loss_ppg_to_gsr + loss_gsr_to_ppg) / 2.0


def retrieval_accuracy(logits: torch.Tensor, k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k retrieval accuracy in both directions for a batch."""

    # labels = torch.arange(logits.size(0), device=logits.device)
    
    # # PPG to GSR: for each PPG (row), check if correct GSR is in top-k predictions
    # _, ppg_topk_indices = logits.topk(k, dim=1)  # (N, k)
    # ppg_correct = torch.zeros_like(labels, dtype=torch.bool)
    # for i in range(logits.size(0)):
    #     ppg_correct[i] = labels[i] in ppg_topk_indices[i]
    # ppg_to_gsr = ppg_correct.float().mean()
    
    # # GSR to PPG: for each GSR (column), check if correct PPG is in top-k predictions
    # _, gsr_topk_indices = logits.topk(k, dim=0)  # (k, N)
    # gsr_correct = torch.zeros_like(labels, dtype=torch.bool)
    # for j in range(logits.size(0)):
    #     gsr_correct[j] = labels[j] in gsr_topk_indices[:, j]
    # gsr_to_ppg = gsr_correct.float().mean()

    labels = torch.arange(logits.size(0), device=logits.device)
    topk = logits.topk(k, dim=1).indices          # (N, k)
    ppg_to_gsr = (topk == labels.unsqueeze(1)).any(dim=1).float().mean()
    topk_t = logits.t().topk(k, dim=1).indices
    gsr_to_ppg = (topk_t == labels.unsqueeze(1)).any(dim=1).float().mean()
    
    return ppg_to_gsr, gsr_to_ppg


def pair_rank_metrics(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate mean and median pair rank in both directions.
    
    For each matched pair (i, i), calculate the rank of the correct match
    in the sorted list of similarities.
    
    Returns:
        ppg_mean_rank: mean rank for PPG to GSR matching
        ppg_median_rank: median rank for PPG to GSR matching
        gsr_mean_rank: mean rank for GSR to PPG matching
        gsr_median_rank: median rank for GSR to PPG matching
    """
    # 
    labels = torch.arange(logits.size(0), device=logits.device)
    ranks_ppg = (logits.argsort(dim=1, descending=True) == labels.unsqueeze(1)).nonzero()[:, 1] + 1
    ranks_gsr = (logits.t().argsort(dim=1, descending=True) == labels.unsqueeze(1)).nonzero()[:, 1] + 1
    return ranks_ppg.float().mean(), ranks_ppg.float().median(), ranks_gsr.float().mean(), ranks_gsr.float().median()
