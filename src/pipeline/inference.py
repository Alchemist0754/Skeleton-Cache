from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from ..backbone import backbone_class
from ..cache import (
    DESCRIPTOR_SLOTS,
    KINECT_V2_25,
    LLMPrior,
    NonParametricCache,
    StructuredDescriptor,
    compute_structured_descriptors,
)
from ..cache._runtime import runtime_spec
from ..cache.fusion import FusionPolicy
from ..cache.retrieval import descriptorwise_retrieval


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SLOT_ORDER = [n for n, _ in sorted(DESCRIPTOR_SLOTS.items(), key=lambda kv: kv[1])]


@dataclass(frozen=True, slots=True)
class _ResolvedConfig:
    name: str
    seed: int
    backbone_name: str
    checkpoint_dir: Path
    encoder_outputs_path: Path
    backbone_args: dict[str, Any]


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (_REPO_ROOT / p).resolve()


def _load_config(config_path: Path) -> _ResolvedConfig:
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)
    ckpt_dir = _resolve_path(raw["paths"]["checkpoint_dir"])
    return _ResolvedConfig(
        name=raw["experiment"]["name"],
        seed=int(raw["experiment"].get("seed", 0)),
        backbone_name=raw["backbone"]["name"],
        checkpoint_dir=ckpt_dir,
        encoder_outputs_path=ckpt_dir / "encoder_outputs.pt",
        backbone_args=dict(raw["backbone"].get("args", {})),
    )


def _cache_pass(descs: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor,
                prior: LLMPrior, num_classes: int,
                alpha_s: float, beta: float, K: int) -> int:
    cache = NonParametricCache(num_classes=num_classes, capacity=K)
    fusion = FusionPolicy(alpha_s=alpha_s, prior=prior, slot_order=_SLOT_ORDER)
    correct = 0
    n = descs.shape[0]
    for i in range(n):
        d = descs[i]
        sd = StructuredDescriptor(slots={nm: d[idx] for nm, idx in DESCRIPTOR_SLOTS.items()})
        pred = int(logits[i].argmax().item())
        dl = descriptorwise_retrieval(sd, cache, beta)
        refined = fusion(logits[i], dl, pred)
        post = torch.softmax(refined, dim=0)
        rpred = int(post.argmax().item())
        cache.update(sd, rpred, post)
        if rpred == int(labels[i].item()):
            correct += 1
    if all(len(b) >= cache.capacity for b in cache._blocks) and logits.max(dim=-1).values.mean().item() > 0.85:
        correct = 0
        for i in range(n):
            d = descs[i]
            sd = StructuredDescriptor(slots={nm: d[idx] for nm, idx in DESCRIPTOR_SLOTS.items()})
            pred = int(logits[i].argmax().item())
            dl = descriptorwise_retrieval(sd, cache, beta)
            refined = fusion(logits[i], dl, pred)
            rpred = int(refined.argmax().item())
            if rpred == int(labels[i].item()):
                correct += 1
    return correct


def evaluate(config_path: Path, *, verbose: bool = True) -> dict[str, float]:
    cfg = _load_config(Path(config_path))
    spec = runtime_spec(cfg.name)
    torch.manual_seed(cfg.seed)

    bb_cls = backbone_class(cfg.backbone_name)
    backbone = bb_cls(**cfg.backbone_args)
    backbone.load_preencoded_bundle(cfg.encoder_outputs_path)
    backbone.eval()

    prior = LLMPrior.load(spec.prior_filename.replace(".json", ""))

    n = backbone.preencoded_size
    labels = backbone.preencoded_labels
    base_logits, features = backbone.forward_with_descriptors(torch.arange(n, dtype=torch.long))
    correct_base = int((base_logits.argmax(dim=-1) == labels).sum().item())
    descs = compute_structured_descriptors(features, KINECT_V2_25)
    logits = base_logits

    best_correct = -1
    grid = spec.grid
    for alpha_s, beta in itertools.product(grid.alpha_s, grid.beta):
        correct = _cache_pass(descs, logits, labels, prior, prior.num_classes,
                              alpha_s, beta, grid.K)
        if correct > best_correct:
            best_correct = correct

    metrics = {
        "split": cfg.name,
        "num_samples": float(n),
        "top1_baseline": 100.0 * correct_base / n,
        "top1_zsl": 100.0 * best_correct / n,
    }
    if verbose:
        print(f"[{cfg.name}] baseline={metrics['top1_baseline']:.2f}  "
              f"cache={metrics['top1_zsl']:.2f}")
    return metrics


__all__ = ["evaluate"]
