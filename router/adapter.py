"""SatQuery AI — multi-head intent adapter.

The only trainable part of the router. Sits on top of the frozen MiniLM
embedding and emits five heads:

    embedding (384)
        |
    LayerNorm
        |
    Linear(384 -> hidden_dim)      default hidden_dim = 128
        |
    GELU
        |
    Dropout
        |
        +--> task_head            Linear(hidden, 6)
        +--> modality_head        Linear(hidden, 4)
        +--> temporal_head        Linear(hidden, 1)   logit
        +--> spatial_head         Linear(hidden, 1)   logit
        +--> language_head        Linear(hidden, 1)   logit

Measured cost (Phase 4 probe): 50,822 parameters, 20 epochs over 4,096 x 384
in 0.28 s on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from router.label_space import (
    BINARY_HEADS,
    NUM_MODALITIES,
    NUM_TASKS,
)


@dataclass
class AdapterOutput:
    """Raw logits from all five heads. No softmax applied here."""

    task_logits: torch.Tensor        # (B, NUM_TASKS)
    modality_logits: torch.Tensor    # (B, NUM_MODALITIES)
    temporal_logit: torch.Tensor     # (B,)
    spatial_logit: torch.Tensor      # (B,)
    language_logit: torch.Tensor     # (B,)

    def binary_logit(self, head: str) -> torch.Tensor:
        if head == "temporal":
            return self.temporal_logit
        if head == "spatial_output":
            return self.spatial_logit
        if head == "language_output":
            return self.language_logit
        raise KeyError(f"unknown binary head: {head!r}")


class IntentAdapter(nn.Module):
    """Multi-head classifier over frozen sentence embeddings."""

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_tasks: int = NUM_TASKS,
        num_modalities: int = NUM_MODALITIES,
    ) -> None:
        super().__init__()

        if input_dim < 1 or hidden_dim < 1:
            raise ValueError(
                f"input_dim and hidden_dim must be positive, got {input_dim}, {hidden_dim}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout
        self.num_tasks = num_tasks
        self.num_modalities = num_modalities

        self.input_norm = nn.LayerNorm(input_dim)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.task_head = nn.Linear(hidden_dim, num_tasks)
        self.modality_head = nn.Linear(hidden_dim, num_modalities)

        # One logit per binary head. P(yes) = sigmoid(logit).
        self.temporal_head = nn.Linear(hidden_dim, 1)
        self.spatial_head = nn.Linear(hidden_dim, 1)
        self.language_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Small-std init on the heads keeps the initial sigmoid near 0.5.

        Without this the binary heads can start saturated, and BCE gradients
        vanish before the task head has learned anything useful.
        """
        for module in (self.task_head, self.modality_head,
                       self.temporal_head, self.spatial_head, self.language_head):
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    # -- forward -----------------------------------------------------------

    def forward(self, embeddings: torch.Tensor) -> AdapterOutput:
        """Args:
            embeddings: (B, input_dim) float tensor. Must already be detached
                from the frozen encoder — the adapter does not back-propagate
                into MiniLM.
        """
        if embeddings.dim() != 2:
            raise ValueError(
                f"expected a 2-D (batch, dim) tensor, got shape {tuple(embeddings.shape)}"
            )
        if embeddings.shape[1] != self.input_dim:
            raise ValueError(
                f"expected input_dim={self.input_dim}, got {embeddings.shape[1]}"
            )

        hidden = self.trunk(self.input_norm(embeddings))

        return AdapterOutput(
            task_logits=self.task_head(hidden),
            modality_logits=self.modality_head(hidden),
            temporal_logit=self.temporal_head(hidden).squeeze(-1),
            spatial_logit=self.spatial_head(hidden).squeeze(-1),
            language_logit=self.language_head(hidden).squeeze(-1),
        )

    # -- serialisation -----------------------------------------------------

    def config_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_p,
            "num_tasks": self.num_tasks,
            "num_modalities": self.num_modalities,
            "binary_heads": list(BINARY_HEADS),
        }

    @classmethod
    def from_config_dict(cls, payload: dict[str, Any]) -> "IntentAdapter":
        return cls(
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            dropout=float(payload["dropout"]),
            num_tasks=int(payload["num_tasks"]),
            num_modalities=int(payload["num_modalities"]),
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"<IntentAdapter in={self.input_dim} hidden={self.hidden_dim} "
            f"params={self.num_parameters()}>"
        )


__all__ = ["IntentAdapter", "AdapterOutput"]