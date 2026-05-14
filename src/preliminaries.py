from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

import torch

Query = TypeVar("Query")
Logits = torch.Tensor
T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True, slots=True)
class SeenUnseenSplit:
    seen: tuple[int, ...]
    unseen: tuple[int, ...]
    class_names: tuple[str, ...]

    def __post_init__(self) -> None:
        overlap = set(self.seen) & set(self.unseen)
        if overlap:
            raise ValueError(f"seen / unseen overlap: {sorted(overlap)}")
        if not self.unseen:
            raise ValueError("unseen set is empty")

    @property
    def num_unseen(self) -> int:
        return len(self.unseen)

    @property
    def num_seen(self) -> int:
        return len(self.seen)


@runtime_checkable
class LabelledDataset(Protocol, Generic[T_co]):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> T_co: ...


@runtime_checkable
class TestTimeAdapter(Protocol):
    def adapt(self, query: torch.Tensor) -> Logits: ...


@runtime_checkable
class FeatureProvider(Protocol):
    DESCRIPTOR_SLOTS: dict[str, int]

    def forward_with_descriptors(self, batch_indices: torch.Tensor): ...

    def register_descriptor_hooks(self, extractor: object) -> None: ...

    def _slot_order(self) -> list[str]: ...


class TrainingFreeAdapter(ABC):
    @abstractmethod
    def _adapt_impl(self, query: torch.Tensor) -> Logits: ...

    # Eq. (2): TF-TTA contract -- adapt must run without gradient updates.
    def adapt(self, query: torch.Tensor) -> Logits:
        with torch.inference_mode():
            out = self._adapt_impl(query)
        if out.requires_grad:
            raise RuntimeError("TrainingFreeAdapter returned a grad-requiring tensor")
        return out


@dataclass(frozen=True, slots=True)
class CacheHyperparameters:
    capacity: int
    alpha_s: float
    beta: float
    prior_split_name: str


__all__ = [
    "SeenUnseenSplit", "LabelledDataset", "TestTimeAdapter", "FeatureProvider",
    "TrainingFreeAdapter", "CacheHyperparameters", "Logits", "Query",
]
