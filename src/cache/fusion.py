from __future__ import annotations

import torch

from .partitioning import DESCRIPTOR_SLOTS
from .prior import LLMPrior


# Eq. (9): w^{(c)} from LLM-derived spatial / temporal / gamma weights, l1-normalized.
def _assemble_weight_vector(prior: LLMPrior, class_index: int,
                            slot_order: list[str]) -> torch.Tensor:
    spa = prior.spatial.get(class_index)
    tmp = prior.temporal.get(class_index)
    gamma = prior.gamma[class_index]
    has_spatial = any(k.startswith("spatial.") for k in slot_order)
    has_temporal = any(k.startswith("temporal.") for k in slot_order)
    if not has_spatial and not has_temporal:
        return torch.tensor([1.0], dtype=torch.float32)
    local_scale = 1.0 - gamma
    raw = [0.0] * len(slot_order)
    spatial_names = [k for k in slot_order if k.startswith("spatial.")]
    temporal_names = [k for k in slot_order if k.startswith("temporal.")]
    for i, name in enumerate(spatial_names):
        raw[slot_order.index(name)] = local_scale * (spa[i] if spa else 1.0 / max(len(spatial_names), 1))
    for i, name in enumerate(temporal_names):
        raw[slot_order.index(name)] = local_scale * (tmp[i] if tmp else 1.0 / max(len(temporal_names), 1))
    if "global" in slot_order:
        raw[slot_order.index("global")] = gamma
    w = torch.tensor(raw, dtype=torch.float32)
    return w / w.sum().clamp_min(1e-12)


# Eq. (10): O = stack of o^{(d)}.
def _stack_descriptor_logits(descriptor_logits: dict[str, torch.Tensor],
                             slot_order: list[str]) -> torch.Tensor:
    return torch.stack([descriptor_logits[name] for name in slot_order], dim=0)


# Eq. (11): s = w O.
def _weighted_sum(weights: torch.Tensor, logits_matrix: torch.Tensor) -> torch.Tensor:
    return weights.unsqueeze(0).mm(logits_matrix).squeeze(0)


# Eq. (12): phi = phi_hat + alpha_s * s.
def fuse(base_logits: torch.Tensor, descriptor_logits: dict[str, torch.Tensor],
         prior: LLMPrior, predicted_class: int, alpha_s: float,
         slot_order: list[str]) -> torch.Tensor:
    w = _assemble_weight_vector(prior, predicted_class, slot_order).to(base_logits.device)
    o_matrix = _stack_descriptor_logits(descriptor_logits, slot_order).to(base_logits.device)
    s = _weighted_sum(w, o_matrix)
    return base_logits + alpha_s * s


class FusionPolicy:
    def __init__(self, alpha_s: float, prior: LLMPrior, slot_order: list[str]) -> None:
        self._alpha_s = alpha_s
        self._prior = prior
        self._slot_order = slot_order

    @property
    def alpha_s(self) -> float:
        return self._alpha_s

    @property
    def prior(self) -> LLMPrior:
        return self._prior

    def __call__(self, base_logits: torch.Tensor,
                 descriptor_logits: dict[str, torch.Tensor],
                 predicted_class: int) -> torch.Tensor:
        return fuse(base_logits, descriptor_logits, self._prior, predicted_class,
                    self._alpha_s, self._slot_order)


__all__ = ["fuse", "FusionPolicy"]
