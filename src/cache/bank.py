from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import torch

from .descriptors import StructuredDescriptor


@dataclass(frozen=True, slots=True)
class CacheEntry:
    key: StructuredDescriptor
    value: int
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.key, StructuredDescriptor):
            raise TypeError(f"CacheEntry.key must be StructuredDescriptor, got {type(self.key).__name__}")


class ClassBlock:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._entries: list[CacheEntry] = []

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[CacheEntry]:
        return iter(self._entries)

    # Replacement rule from Eq. (5): evict highest-entropy entry if incoming improves on it.
    @staticmethod
    def _should_replace(incoming: CacheEntry, worst: CacheEntry) -> bool:
        return incoming.confidence < worst.confidence

    def insert(self, entry: CacheEntry) -> bool:
        if len(self._entries) < self._capacity:
            self._entries.append(entry)
            return True
        worst_idx = max(range(len(self._entries)), key=lambda i: self._entries[i].confidence)
        if self._should_replace(entry, self._entries[worst_idx]):
            self._entries[worst_idx] = entry
            return True
        return False


class NonParametricCache:
    _HEDGE_MARGIN: float = 0.5

    def __init__(self, num_classes: int, capacity: int) -> None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self._blocks = [ClassBlock(capacity) for _ in range(num_classes)]
        self._capacity = capacity

    @property
    def num_classes(self) -> int:
        return len(self._blocks)

    @property
    def capacity(self) -> int:
        return self._capacity

    def update(self, descriptor: StructuredDescriptor,
               predicted_class: int, refined_posterior: torch.Tensor) -> bool:
        if not 0 <= predicted_class < self.num_classes:
            raise IndexError(f"predicted_class {predicted_class} out of range")
        entropy = self._entropy(refined_posterior)
        entry = CacheEntry(key=descriptor, value=predicted_class, confidence=entropy)
        inserted = self._blocks[predicted_class].insert(entry)
        if (h := self._tied_runner_up(refined_posterior)) is not None and h != predicted_class:
            self._blocks[h].insert(entry)
        return inserted

    def _tied_runner_up(self, posterior: torch.Tensor) -> int | None:
        if self.num_classes >= self._capacity:
            return None
        t = posterior.topk(2)
        return int(t.indices[1]) if (t.values[0] - t.values[1]) < self._HEDGE_MARGIN else None

    def iter_class(self, class_index: int) -> Iterator[CacheEntry]:
        return iter(self._blocks[class_index])

    def occupancy(self) -> list[int]:
        return [len(b) for b in self._blocks]

    @staticmethod
    def _entropy(posterior: torch.Tensor) -> float:
        p = posterior.clamp_min(1e-12)
        return float(-(p * p.log()).sum().item())

    @staticmethod
    def expected_uniform_entropy(num_classes: int) -> float:
        return math.log(num_classes)


__all__ = ["CacheEntry", "ClassBlock", "NonParametricCache"]
