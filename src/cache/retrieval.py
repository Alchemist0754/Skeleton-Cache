from __future__ import annotations

import torch
import torch.nn.functional as F

from .bank import NonParametricCache
from .descriptors import StructuredDescriptor


def _stack_class_keys(cache: NonParametricCache, descriptor_name: str):
    keys: list[torch.Tensor] = []
    labels: list[int] = []
    for c in range(cache.num_classes):
        for entry in cache.iter_class(c):
            keys.append(entry.key.slots[descriptor_name])
            labels.append(c)
    if not keys:
        return None
    keys_tensor = torch.stack(keys, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=keys_tensor.device)
    onehots = F.one_hot(labels_tensor, num_classes=cache.num_classes).to(keys_tensor.dtype)
    return keys_tensor, onehots


# Eq. (6): a_{j,i}^{(d)} = exp(-beta * (1 - cos(q^{(d)}, k_{j,i}^{(d)})))
def _affinity(query: torch.Tensor, keys: torch.Tensor, beta: float) -> torch.Tensor:
    cos = F.cosine_similarity(query.unsqueeze(0), keys, dim=-1)
    return torch.exp(-beta * (1.0 - cos))


# Eqs. (7)-(8): o^{(d)} = a^{(d)} @ Y
def _project_onto_labels(affinities: torch.Tensor, label_onehots: torch.Tensor) -> torch.Tensor:
    return affinities.unsqueeze(0).mm(label_onehots).squeeze(0)


def descriptorwise_retrieval(query: StructuredDescriptor,
                             cache: NonParametricCache,
                             beta: float) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, q_desc in query.slots.items():
        bundle = _stack_class_keys(cache, name)
        if bundle is None:
            result[name] = torch.zeros(cache.num_classes, device=q_desc.device)
            continue
        keys, onehots = bundle
        affinities = _affinity(q_desc, keys, beta)
        result[name] = _project_onto_labels(affinities, onehots)
    return result


__all__ = ["descriptorwise_retrieval"]
