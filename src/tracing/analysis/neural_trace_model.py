#!/usr/bin/env python3
"""Small dependency-optional neural prefix encoder for trace prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:  # Keep preprocessing and CLI inspection usable on machines without torch.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised on the current CPU-only laptop.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ModelConfig:
    node_vocab_size: int
    domain_vocab_size: int
    planner_vocab_size: int
    label_size: int
    hidden_size: int = 256
    num_layers: int = 2
    max_position: int = 64
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError(
            "PyTorch is required for neural training; preprocessing remains dependency-free. "
            "Install a CUDA-compatible torch build on the remote GPU environment."
        )


if nn is not None:

    class TracePrefixPredictor(nn.Module):
        """GRU prefix encoder with next-node, END and length heads.

        The model intentionally has no public-dataset action vocabulary built
        in.  The caller supplies the vocabulary for the current domain.  A
        train-only graph mask can be applied to ``next_logits`` before the
        softmax by the training/evaluation adapter.
        """

        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            self.config = config
            hidden = config.hidden_size
            self.node_embedding = nn.Embedding(config.node_vocab_size, hidden, padding_idx=0)
            self.domain_embedding = nn.Embedding(config.domain_vocab_size, hidden)
            self.planner_embedding = nn.Embedding(config.planner_vocab_size, hidden)
            self.position_embedding = nn.Embedding(config.max_position + 1, hidden)
            self.encoder = nn.GRU(
                input_size=hidden,
                hidden_size=hidden,
                num_layers=config.num_layers,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(hidden * 4)
            self.dropout = nn.Dropout(config.dropout)
            self.next_head = nn.Linear(hidden * 4, config.label_size)
            self.end_head = nn.Linear(hidden * 4, 1)
            self.length_head = nn.Linear(hidden * 4, 1)

        def forward(
            self,
            node_ids: Any,
            lengths: Any,
            domain_ids: Any,
            planner_ids: Any,
            positions: Any,
        ) -> dict[str, Any]:
            require_torch()
            embedded = self.node_embedding(node_ids)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.encoder(packed)
            sequence_state = hidden[-1]
            position_ids = positions.clamp(min=0, max=self.config.max_position)
            fused = torch.cat(
                (
                    sequence_state,
                    self.domain_embedding(domain_ids),
                    self.planner_embedding(planner_ids),
                    self.position_embedding(position_ids),
                ),
                dim=-1,
            )
            fused = self.dropout(self.norm(fused))
            return {
                "state": fused,
                "next_logits": self.next_head(fused),
                "end_logits": self.end_head(fused).squeeze(-1),
                "length_pred": self.length_head(fused).squeeze(-1),
            }

else:

    class TracePrefixPredictor:  # pragma: no cover - only used without torch.
        def __init__(self, config: ModelConfig) -> None:
            del config
            require_torch()

