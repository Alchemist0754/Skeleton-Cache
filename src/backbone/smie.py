from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BackboneAdapter
from ..cache.partitioning import DESCRIPTOR_SLOTS


SMIE_DESCRIPTOR_SLOTS: Final[dict[str, int]] = DESCRIPTOR_SLOTS


class _GlobalDiscriminator(nn.Module):
    def __init__(self, in_feature: int) -> None:
        super().__init__()
        self.l0 = nn.Linear(in_feature, 1024)
        self.l1 = nn.Linear(1024, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, visual: torch.Tensor, language: torch.Tensor) -> torch.Tensor:
        x = torch.cat((visual, language), dim=-1)
        x = F.relu(self.l0(x))
        x = F.relu(self.l1(x))
        return self.l2(x)


class _MI(nn.Module):
    def __init__(self, visual_size: int, language_size: int) -> None:
        super().__init__()
        self.global_D = _GlobalDiscriminator(visual_size + language_size)
        self.ln = nn.LayerNorm([visual_size], elementwise_affine=False)


class SmieBackbone(BackboneAdapter):
    DESCRIPTOR_SLOTS: Final[dict[str, int]] = SMIE_DESCRIPTOR_SLOTS

    def __init__(self, visual_size: int = 256, language_size: int = 768,
                 stgcn_repo: str | None = None, stgcn_ckpt: str | None = None) -> None:
        super().__init__()
        self.visual_size = visual_size
        self.language_size = language_size
        self.mi = _MI(visual_size, language_size)
        self._stgcn = None
        self._stgcn_repo = stgcn_repo
        self._stgcn_ckpt = stgcn_ckpt
        self._unseen_indices: list[int] | None = None
        self._unseen_text: torch.Tensor | None = None
        self._pre_pool_buffer: dict[str, torch.Tensor] = {}

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path,
                        backbone_args: dict | None = None) -> "SmieBackbone":
        model = cls(**(backbone_args or {}))
        if checkpoint_path.is_file():
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if hasattr(state, "state_dict"):
                sd = state.state_dict()
            elif isinstance(state, dict) and "state_dict" in state:
                sd = state["state_dict"]
            else:
                sd = state
            cleaned = {("mi." + k if not k.startswith("mi.") else k): v for k, v in sd.items()}
            own_keys = set(model.state_dict().keys())
            filtered = {k: v for k, v in cleaned.items() if k in own_keys}
            model.load_state_dict(filtered, strict=False)
        model.eval()
        return model

    def register_descriptor_hooks(self, extractor: "object") -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []
        del extractor

    def attach_stgcn(self, stgcn_repo_path: str, ckpt_path: str) -> None:
        if stgcn_repo_path not in sys.path:
            sys.path.insert(0, stgcn_repo_path)
        from module.gcn.st_gcn import Model as STGCN
        graph_args = {"layout": "ntu-rgb+d", "strategy": "spatial"}
        encoder = STGCN(in_channels=3, hidden_channels=16, hidden_dim=256,
                        dropout=0.5, graph_args=graph_args,
                        edge_importance_weighting=True)
        encoder.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
        encoder.eval()
        self._stgcn = encoder

        def _hook(_m, _i, output):
            self._pre_pool_buffer["x"] = output if isinstance(output, torch.Tensor) else output[0]
        encoder.st_gcn_networks[-1].register_forward_hook(_hook)

    def set_unseen_context(self, unseen_class_indices: list[int], unseen_text: torch.Tensor) -> None:
        self._unseen_indices = list(unseen_class_indices)
        self._unseen_text = unseen_text

    def _forward_from_skeletons(self, batch: torch.Tensor):
        if self._stgcn is None:
            raise NotImplementedError(
                "SmieBackbone needs attach_stgcn(repo_path, ckpt) before from-skeletons forward."
            )
        if self._unseen_text is None:
            raise RuntimeError("Call set_unseen_context(unseen_indices, unseen_text) first")
        from einops import repeat
        device = next(self._stgcn.parameters()).device
        x = batch.to(device)
        _ = self._stgcn(x)
        pre = self._pre_pool_buffer["x"]
        B = x.shape[0]
        M = x.shape[-1]
        pre = pre.view(B, M, *pre.shape[1:]).mean(dim=1)  # (B, 256, T, V)
        global_feat = pre.mean(dim=(-1, -2))              # (B, 256)
        n_unseen = self._unseen_text.shape[0]
        v = repeat(global_feat, "b c -> b u c", u=n_unseen)
        l = repeat(self._unseen_text.to(device), "u c -> b u c", b=B)
        v = self.mi.ln(v)
        scores = -F.softplus(-self.mi.global_D(v, l)).squeeze(-1)
        return scores, pre


__all__ = ["SmieBackbone", "SMIE_DESCRIPTOR_SLOTS"]
