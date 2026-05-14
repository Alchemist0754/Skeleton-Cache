from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_DATA_DIR = Path(__file__).resolve().parent / "data"


def _validate_distribution(name: str, values: list[float], length: int) -> list[float]:
    if len(values) != length:
        raise ValueError(f"{name}: expected length {length}, got {len(values)}")
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError(f"{name}: entries must be finite and non-negative")
    total = sum(values)
    if total <= 0:
        raise ValueError(f"{name}: all-zero distribution is not permitted")
    return [v / total for v in values]


@dataclass(frozen=True, slots=True)
class LLMPrior:
    split: str
    num_classes: int
    class_names: tuple[str, ...]
    spatial: dict[int, tuple[float, ...]]
    temporal: dict[int, tuple[float, ...]]
    gamma: dict[int, float]
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, split: str) -> "LLMPrior":
        path = _DATA_DIR / f"{split}.json"
        if not path.is_file():
            raise FileNotFoundError(f"LLM prior file for split {split!r} not found at {path}")
        return cls._from_raw(split, json.loads(path.read_text()))

    @classmethod
    def _from_raw(cls, split: str, raw: dict[str, Any]) -> "LLMPrior":
        if raw.get("split") != split:
            raise ValueError(f"prior file split {raw.get('split')!r} != expected {split!r}")
        num_classes = int(raw["num_classes"])
        class_names = tuple(raw.get("class_names", [f"class_{i}" for i in range(num_classes)]))
        if len(class_names) != num_classes:
            raise ValueError("class_names length must equal num_classes")
        weights_raw = raw["weights"]
        if len(weights_raw) != num_classes:
            raise ValueError(f"expected {num_classes} entries in 'weights', got {len(weights_raw)}")
        spatial: dict[int, tuple[float, ...]] = {}
        temporal: dict[int, tuple[float, ...]] = {}
        gamma: dict[int, float] = {}
        for key, payload in weights_raw.items():
            idx = int(key)
            if not 0 <= idx < num_classes:
                raise ValueError(f"class index {idx} out of range [0, {num_classes})")
            if "spatial" in payload:
                spa = _validate_distribution(f"weights[{idx}].spatial", list(payload["spatial"]), 4)
                spatial[idx] = tuple(spa)
            if "temporal" in payload:
                tmp = _validate_distribution(f"weights[{idx}].temporal", list(payload["temporal"]), 3)
                temporal[idx] = tuple(tmp)
            g = float(payload["gamma"])
            if not 0.0 <= g <= 1.0:
                raise ValueError(f"weights[{idx}].gamma must lie in [0, 1]; got {g}")
            gamma[idx] = g
        missing = sorted(set(range(num_classes)) - set(gamma))
        if missing:
            raise ValueError(f"prior file is missing entries for class indices {missing}")
        meta = {k: v for k, v in raw.items()
                if k not in {"split", "num_classes", "class_names", "weights"}}
        return cls(split=split, num_classes=num_classes, class_names=class_names,
                   spatial=spatial, temporal=temporal, gamma=gamma, raw_metadata=meta)


__all__ = ["LLMPrior"]
