from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Final, MutableMapping

import torch
import torch.nn as nn


_BUNDLE_SCHEMA_VERSION: Final[int] = 2


class BackboneAdapter(nn.Module, ABC):
    DESCRIPTOR_SLOTS: dict[str, int]

    def __init__(self) -> None:
        super().__init__()
        self._preencoded: dict[str, torch.Tensor] | None = None
        self._descriptor_taps: MutableMapping[str, torch.Tensor] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []

    @classmethod
    @abstractmethod
    def from_checkpoint(cls, checkpoint_path: Path, backbone_args: dict | None = None) -> "BackboneAdapter":
        ...

    @abstractmethod
    def register_descriptor_hooks(self, extractor: "object") -> None: ...

    def consume_taps(self) -> dict[str, torch.Tensor]:
        out = dict(self._descriptor_taps)
        self._descriptor_taps.clear()
        return out

    def load_preencoded_bundle(self, bundle_path: Path) -> None:
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        version = int(bundle.get("schema_version", 0))
        if version != _BUNDLE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported bundle schema {version} != {_BUNDLE_SCHEMA_VERSION}"
            )
        if "skeleton_features" not in bundle:
            raise RuntimeError("Bundle missing 'skeleton_features' (the encoder F volume)")
        self._preencoded = {
            "skeleton_features": bundle["skeleton_features"],
            "logits": bundle["logits"],
            "labels": bundle["labels"],
            "num_unseen": int(bundle.get("num_unseen", bundle["logits"].shape[-1])),
        }

    def _slot_order(self) -> list[str]:
        return [name for name, _ in sorted(self.DESCRIPTOR_SLOTS.items(), key=lambda kv: kv[1])]

    @property
    def num_slots(self) -> int:
        return len(self.DESCRIPTOR_SLOTS)

    @property
    def preencoded_size(self) -> int:
        if self._preencoded is None:
            return 0
        return int(self._preencoded["labels"].shape[0])

    @property
    def preencoded_labels(self) -> torch.Tensor:
        if self._preencoded is None:
            raise RuntimeError("No pre-encoded bundle is loaded.")
        return self._preencoded["labels"]

    @torch.inference_mode()
    def forward_with_descriptors(self, batch_indices: torch.Tensor):
        if self._preencoded is not None:
            return self._dispatch_preencoded(batch_indices)
        return self._forward_from_skeletons(batch_indices)

    def _dispatch_preencoded(self, batch_indices: torch.Tensor):
        assert self._preencoded is not None
        idx = batch_indices.to(self._preencoded["labels"].device).long()
        logits = self._preencoded["logits"].index_select(0, idx)
        features = self._preencoded["skeleton_features"].index_select(0, idx)
        return logits, features

    @abstractmethod
    def _forward_from_skeletons(self, batch: torch.Tensor): ...


__all__ = ["BackboneAdapter"]
