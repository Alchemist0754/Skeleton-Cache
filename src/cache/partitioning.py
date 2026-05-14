from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch


DESCRIPTOR_SLOTS: Final[dict[str, int]] = {
    "spatial.head": 0,
    "spatial.torso": 1,
    "spatial.arms": 2,
    "spatial.feet": 3,
    "temporal.begin": 4,
    "temporal.middle": 5,
    "temporal.end": 6,
    "global": 7,
}


@dataclass(frozen=True, slots=True)
class SkeletonPartitioning:
    spatial_groups: dict[str, tuple[int, ...]]
    temporal_phases: tuple[tuple[float, float], ...]

    @property
    def num_spatial(self) -> int:
        return len(self.spatial_groups)

    @property
    def num_temporal(self) -> int:
        return len(self.temporal_phases)


# Section 4.2 joint groups for the NTU Kinect-v2 25-joint skeleton.
KINECT_V2_25 = SkeletonPartitioning(
    spatial_groups={
        "head":  (2, 3, 4, 8, 20),
        "torso": (0, 1, 4, 8, 12, 16, 20),
        "arms":  (4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23, 24),
        "feet":  (0, 12, 13, 14, 15, 16, 17, 18, 19),
    },
    temporal_phases=((0.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0)),
)


# Eq. (3): s_p, t_z, g averages from the latent tensor F.
def compute_structured_descriptors(features: torch.Tensor,
                                   partitioning: SkeletonPartitioning = KINECT_V2_25) -> torch.Tensor:
    if features.dim() != 4:
        raise ValueError(f"expected (B, C, T, V), got shape {tuple(features.shape)}")
    B, C, T, V = features.shape
    out = torch.empty(B, len(DESCRIPTOR_SLOTS), C,
                      dtype=features.dtype, device=features.device)
    for part, joints in partitioning.spatial_groups.items():
        idx = torch.tensor(joints, dtype=torch.long, device=features.device)
        g = features.index_select(-1, idx)
        out[:, DESCRIPTOR_SLOTS[f"spatial.{part}"]] = g.mean(dim=(-1, -2))
    edges = [int(round(T * lo)) for lo, _ in partitioning.temporal_phases] + [T]
    for i, key in enumerate(("temporal.begin", "temporal.middle", "temporal.end")):
        s, e = edges[i], max(edges[i + 1], edges[i] + 1)
        out[:, DESCRIPTOR_SLOTS[key]] = features[:, :, s:e, :].mean(dim=(-1, -2))
    out[:, DESCRIPTOR_SLOTS["global"]] = features.mean(dim=(-1, -2))
    return out


__all__ = ["SkeletonPartitioning", "KINECT_V2_25", "DESCRIPTOR_SLOTS",
           "compute_structured_descriptors"]
