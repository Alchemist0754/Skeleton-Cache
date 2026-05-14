from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from ..preliminaries import FeatureProvider
from .partitioning import (
    DESCRIPTOR_SLOTS,
    KINECT_V2_25,
    SkeletonPartitioning,
    compute_structured_descriptors,
)


@dataclass(frozen=True, slots=True)
class StructuredDescriptor:
    slots: dict[str, torch.Tensor]

    def as_stack(self, slot_order: list[str]) -> torch.Tensor:
        return torch.stack([self.slots[k] for k in slot_order], dim=0)

    def named_iter(self) -> Iterable[tuple[str, torch.Tensor]]:
        return iter(self.slots.items())


class DescriptorExtractor:
    def __init__(self, backbone: FeatureProvider,
                 partitioning: SkeletonPartitioning | None = None) -> None:
        for attr in ("DESCRIPTOR_SLOTS", "forward_with_descriptors"):
            if not hasattr(backbone, attr):
                raise TypeError(f"backbone must satisfy FeatureProvider; missing {attr!r}")
        self._backbone = backbone
        self._slots = backbone.DESCRIPTOR_SLOTS
        self._partitioning = partitioning if partitioning is not None else KINECT_V2_25
        if hasattr(backbone, "register_descriptor_hooks"):
            backbone.register_descriptor_hooks(self)

    def extract(self, batch_indices: torch.Tensor):
        logits, features = self._backbone.forward_with_descriptors(batch_indices)
        descriptor_stack = compute_structured_descriptors(features, self._partitioning)
        descs = [self._unpack(descriptor_stack[i]) for i in range(descriptor_stack.shape[0])]
        return logits, descs

    def _unpack(self, stack: torch.Tensor) -> StructuredDescriptor:
        return StructuredDescriptor(slots={name: stack[idx] for name, idx in self._slots.items()})


__all__ = ["DescriptorExtractor", "StructuredDescriptor"]
