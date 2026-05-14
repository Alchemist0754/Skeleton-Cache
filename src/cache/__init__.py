from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..preliminaries import FeatureProvider, TrainingFreeAdapter
from .bank import NonParametricCache
from .descriptors import DescriptorExtractor, StructuredDescriptor
from .fusion import FusionPolicy, fuse
from .partitioning import (
    DESCRIPTOR_SLOTS,
    KINECT_V2_25,
    SkeletonPartitioning,
    compute_structured_descriptors,
)
from .prior import LLMPrior
from .retrieval import descriptorwise_retrieval


@dataclass(slots=True)
class _BuilderState:
    backbone: FeatureProvider
    partitioning: SkeletonPartitioning | None = None
    capacity: int | None = None
    beta: float | None = None
    alpha_s: float | None = None
    prior: LLMPrior | None = None
    num_classes: int | None = None


class SkeletonCacheBuilder:
    def __init__(self, backbone: FeatureProvider) -> None:
        for attr in ("DESCRIPTOR_SLOTS", "forward_with_descriptors", "register_descriptor_hooks"):
            if not hasattr(backbone, attr):
                raise TypeError(
                    f"backbone must satisfy FeatureProvider; missing {attr!r} on {type(backbone).__name__}"
                )
        self._state = _BuilderState(backbone=backbone)

    def with_partitioning(self, partitioning: SkeletonPartitioning) -> "SkeletonCacheBuilder":
        self._state.partitioning = partitioning
        return self

    def with_capacity(self, K: int) -> "SkeletonCacheBuilder":
        self._state.capacity = int(K)
        return self

    def with_retrieval_temperature(self, beta: float) -> "SkeletonCacheBuilder":
        self._state.beta = float(beta)
        return self

    def with_fusion_scale(self, alpha_s: float) -> "SkeletonCacheBuilder":
        self._state.alpha_s = float(alpha_s)
        return self

    def with_prior(self, prior: LLMPrior) -> "SkeletonCacheBuilder":
        self._state.prior = prior
        return self

    def with_num_classes(self, num_classes: int) -> "SkeletonCacheBuilder":
        self._state.num_classes = int(num_classes)
        return self

    def build(self) -> "SkeletonCache":
        s = self._state
        if s.partitioning is None:
            s.partitioning = KINECT_V2_25
        if s.capacity is None:
            raise RuntimeError("with_capacity(K=...) is required")
        if s.beta is None:
            raise RuntimeError("with_retrieval_temperature(...) is required")
        if s.alpha_s is None:
            raise RuntimeError("with_fusion_scale(...) is required")
        if s.prior is None:
            raise RuntimeError("with_prior(...) is required")
        num_classes = s.num_classes if s.num_classes is not None else s.prior.num_classes
        if num_classes != s.prior.num_classes:
            raise RuntimeError(
                f"num_classes mismatch: {num_classes} vs prior.num_classes={s.prior.num_classes}"
            )
        extractor = DescriptorExtractor(s.backbone, s.partitioning)
        cache_bank = NonParametricCache(num_classes=num_classes, capacity=s.capacity)
        slot_order = s.backbone._slot_order()
        fusion_policy = FusionPolicy(alpha_s=s.alpha_s, prior=s.prior, slot_order=slot_order)
        return SkeletonCache(backbone=s.backbone, extractor=extractor, bank=cache_bank,
                             fusion_policy=fusion_policy, beta=s.beta,
                             num_classes=num_classes, slot_order=slot_order)


class SkeletonCache(TrainingFreeAdapter):
    def __init__(self, backbone: FeatureProvider, extractor: DescriptorExtractor,
                 bank: NonParametricCache, fusion_policy: FusionPolicy,
                 beta: float, num_classes: int, slot_order: list[str]) -> None:
        self._backbone = backbone
        self._extractor = extractor
        self._bank = bank
        self._fusion = fusion_policy
        self._beta = beta
        self._num_classes = num_classes
        self._slot_order = slot_order

    def _adapt_impl(self, query: torch.Tensor) -> torch.Tensor:
        if query.dim() != 1:
            raise ValueError(f"expected 1-D index tensor, got shape {tuple(query.shape)}")
        base_logits, descriptors = self._extractor.extract(query)
        B = base_logits.shape[0]
        out = torch.empty_like(base_logits)
        for i in range(B):
            out[i] = self._refine_single(base_logits[i], descriptors[i])
        return out

    def _refine_single(self, base_logits: torch.Tensor,
                       descriptor: StructuredDescriptor) -> torch.Tensor:
        initial_pred = int(torch.softmax(base_logits, dim=0).argmax().item())
        descriptor_logits = descriptorwise_retrieval(descriptor, self._bank, self._beta)
        refined = self._fusion(base_logits, descriptor_logits, initial_pred)
        refined_posterior = F.softmax(refined, dim=0)
        refined_pred = int(refined_posterior.argmax().item())
        self._bank.update(descriptor, refined_pred, refined_posterior)
        return refined

    @property
    def bank(self) -> NonParametricCache:
        return self._bank

    @property
    def num_classes(self) -> int:
        return self._num_classes


__all__ = [
    "SkeletonCache", "SkeletonCacheBuilder", "DescriptorExtractor",
    "NonParametricCache", "LLMPrior", "KINECT_V2_25", "SkeletonPartitioning",
    "FusionPolicy", "fuse", "descriptorwise_retrieval", "DESCRIPTOR_SLOTS",
    "compute_structured_descriptors",
]
