from __future__ import annotations

from pathlib import Path
from typing import Final

import torch
import torch.nn as nn

from .base import BackboneAdapter
from ..cache.partitioning import DESCRIPTOR_SLOTS


class SaDvaeBackbone(BackboneAdapter):
    DESCRIPTOR_SLOTS: Final[dict[str, int]] = DESCRIPTOR_SLOTS

    def __init__(self, **_) -> None:
        super().__init__()
        self._marker = nn.Parameter(torch.zeros(1), requires_grad=False)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path,
                        backbone_args: dict | None = None) -> "SaDvaeBackbone":
        return cls(**(backbone_args or {}))

    def register_descriptor_hooks(self, extractor: "object") -> None:
        del extractor

    def _forward_from_skeletons(self, batch: torch.Tensor):
        raise NotImplementedError("SaDvaeBackbone: provide a pre-encoded bundle.")


__all__ = ["SaDvaeBackbone"]
