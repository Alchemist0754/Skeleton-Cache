from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class _SearchGrid:
    K: int = 8
    alpha_s: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0)
    beta: tuple[float, ...] = (3.0, 5.0, 7.0, 9.0)


@dataclass(frozen=True, slots=True)
class _RuntimeSpec:
    prior_filename: str
    grid: _SearchGrid = field(default_factory=_SearchGrid)


_TABLE: dict[str, _RuntimeSpec] = {
    "ntu60_55_5":    _RuntimeSpec(prior_filename="ntu60_55_5.json"),
    "ntu60_48_12":   _RuntimeSpec(prior_filename="ntu60_48_12.json"),
    "ntu120_110_10": _RuntimeSpec(prior_filename="ntu120_110_10.json"),
    "ntu120_96_24":  _RuntimeSpec(prior_filename="ntu120_96_24.json"),
}


def runtime_spec(split_name: str) -> _RuntimeSpec:
    if split_name not in _TABLE:
        raise KeyError(
            f"No runtime spec registered for split {split_name!r}; "
            f"available: {sorted(_TABLE)}"
        )
    return _TABLE[split_name]


__all__ = ["runtime_spec"]
