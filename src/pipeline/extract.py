from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ..backbone.smie import SmieBackbone


_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUNDLE_SCHEMA_VERSION = 2
_SLOT_ORDER = ["spatial.head", "spatial.torso", "spatial.arms", "spatial.feet",
               "temporal.begin", "temporal.middle", "temporal.end", "global"]


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (_REPO_ROOT / pp).resolve()


def extract_smie_bundle(config_path: Path) -> Path:
    with Path(config_path).open("r") as f:
        cfg = yaml.safe_load(f)
    exp = cfg["experiment"]
    bb_cfg = cfg["backbone"]
    paths = cfg["paths"]

    test_data = np.load(_resolve(paths["test_data"]))
    test_label = np.load(_resolve(paths["test_label"]))
    full_language = np.load(_resolve(paths["language_embeddings"]))

    unseen_label = sorted(set(int(x) for x in test_label))
    label_map = {orig: i for i, orig in enumerate(unseen_label)}
    remap = torch.tensor([label_map[int(l)] for l in test_label], dtype=torch.long)

    stgcn_repo = str(_resolve(paths["stgcn_repo"]))
    if stgcn_repo not in sys.path:
        sys.path.insert(0, stgcn_repo)
    bb = SmieBackbone.from_checkpoint(
        _resolve(paths["mi_checkpoint"]),
        backbone_args={"visual_size": int(bb_cfg["args"]["visual_size"]),
                       "language_size": int(bb_cfg["args"]["language_size"])},
    )
    bb.attach_stgcn(stgcn_repo, str(_resolve(paths["stgcn_checkpoint"])))

    lang = torch.tensor(full_language)
    lang = F.normalize(lang, dim=-1)
    unseen_text = lang[unseen_label]
    bb.set_unseen_context(unseen_label, unseen_text)
    bb.to("cuda" if torch.cuda.is_available() else "cpu")

    F_list: list[torch.Tensor] = []
    logits_list: list[torch.Tensor] = []
    bs = int(cfg["extract"].get("batch_size", 32))
    n = len(test_data)
    with torch.no_grad():
        for i in range(0, n, bs):
            xb = torch.tensor(test_data[i:i+bs], dtype=torch.float32)
            scores, features = bb._forward_from_skeletons(xb)
            F_list.append(features.cpu())
            logits_list.append(scores.cpu())

    features = torch.cat(F_list, dim=0).float()
    logits = torch.cat(logits_list, dim=0).float()

    bundle = {
        "schema_version": _BUNDLE_SCHEMA_VERSION,
        "backbone": "smie",
        "dataset": exp["dataset"],
        "split": exp["name"],
        "num_unseen": len(unseen_label),
        "unseen_class_indices": unseen_label,
        "slot_order": _SLOT_ORDER,
        "skeleton_features": features,
        "logits": logits,
        "labels": remap,
    }
    out_dir = _resolve(paths["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "encoder_outputs.pt"
    torch.save(bundle, out_path)
    return out_path


__all__ = ["extract_smie_bundle"]
