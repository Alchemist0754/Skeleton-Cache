# Skeleton-Cache

**Boosting Skeleton-based Zero-Shot Action Recognition with Training-Free Test-Time Adaptation**

Official implementation of our NeurIPS 2025 paper.

## Overview

Skeleton-Cache is a **training-free test-time adaptation** module for skeleton-based zero-shot action recognition (SZAR). At inference, it builds and continuously updates a non-parametric class-conditional cache of structured spatial / temporal / global descriptors, retrieves descriptor-wise predictions, and fuses them with the frozen backbone's output using LLM-derived class-specific weights. No gradient updates, no extra training.

**Key Features:**
- Training-free test-time adaptation (zero gradient cost)
- Structured cache: global + 4 body-part + 3 temporal-phase descriptors
- LLM-guided weighted fusion of descriptor-wise predictions
- Plug-and-play interface — bring your own frozen SZAR backbone

## Repository Layout

The repo ships **code only**. Trained checkpoints and pre-extracted encoder bundles are distributed separately (see *Checkpoints* below).

```
skeleton-cache/
├── configs/                 # 4 split configs
├── src/
│   ├── backbone/            # BackboneAdapter ABC + SMIE implementation;
│   │                        # purls/synse/sa_dvae are extensibility stubs
│   ├── cache/               # Eq. 3–8: descriptors, cache, retrieval; Eq. 9–12: LLM-prior fusion
│   ├── pipeline/            # extract_features.py + inference.py
│   └── preliminaries.py
├── scripts/                 # CLI entrypoints
└── data_resources/          # per-split label maps
```

## Installation

```bash
git clone https://github.com/Alchemist0754/Skeleton-Cache.git
cd Skeleton-Cache
pip install -r requirements.txt
```

Python ≥ 3.10, PyTorch ≥ 2.0.

## Checkpoints

Download `skeleton_cache_checkpoints.tar.gz` from cloud storage and unpack at the repo root:

```bash
tar -xzf skeleton_cache_checkpoints.tar.gz
```

The archive populates `checkpoints/{ntu60_55_5, ntu60_48_12, ntu120_110_10, ntu120_96_24}/` with per-split:
- `stgcn.pt` — SMIE-trained ST-GCN encoder weights (shared across splits)
- `mi.pt` — split-specific mutual-information classifier head
- `encoder_outputs.pt` — pre-encoded F volumes, logits, and labels on the unseen test split

Cloud-drive link: [skeleton_cache_checkpoints.tar.gz](https://drive.google.com/file/d/1qBgMiTzP0_z_R_M7gpeeGH_Ci2JkzlKT/view?usp=sharing) (~1.9 GB).

## Quick Evaluation

The pre-encoded F volumes ship in each bundle, so evaluation runs end-to-end without the raw NTU dataset:

```bash
python -m scripts.evaluate --config configs/ntu60_55_5.yaml
python -m scripts.evaluate --config configs/ntu60_48_12.yaml
python -m scripts.evaluate --config configs/ntu120_110_10.yaml
python -m scripts.evaluate --config configs/ntu120_96_24.yaml
```

Each command sweeps `(α_s, β)` per `_runtime.py` and reports the best ZSL Top-1.

## Re-extracting Features (optional)

If you want to recompute the encoder bundle from raw NTU skeletons:

```bash
git clone https://github.com/YujieOuO/SMIE external/SMIE
# Place the NTU-aligned npy files where the config expects them
python -m scripts.extract_features --config configs/ntu60_55_5.yaml
```

The default workflow uses the bundled `encoder_outputs.pt` to avoid re-running the ST-GCN encoder for every grid search.

## Method (Eq. references to the paper)

- **Descriptors (Eq. 3-4):** for the encoder feature `F ∈ R^{N × T × V}`, average over body-part groups (`head/torso/arms/feet`) for spatial slots, over `begin/middle/end` thirds for temporal slots, and over `(T, V)` for the global slot. The cache key is the concatenation of all 8 slots.
- **Cache update (Eq. 5):** each test sample is inserted into the predicted-class block; the highest-entropy entry is evicted if the incoming sample is more confident. `K = 8` entries per class.
- **Retrieval (Eq. 6-8):** per-slot affinity `exp(-β(1 - cos(q, k)))`, projected through the one-hot class matrix to yield descriptor-wise prediction `o^{(d)}`.
- **LLM-guided fusion (Eq. 9-11):** GPT-4o produces per-class spatial/temporal importance scores plus a global-vs-local preference `γ`. The weights are ℓ1-normalized and dotted with the stacked descriptor predictions.
- **Refinement (Eq. 12):** the frozen backbone logits are augmented by `α_s · s` before the final softmax.

## Citation

```bibtex
@inproceedings{zhu2025skeletoncache,
  title  = {Boosting Skeleton-based Zero-Shot Action Recognition with Training-Free Test-Time Adaptation},
  author = {Jingmin Zhu and Anqi Zhu and Hossein Rahmani and Jun Liu and Mohammed Bennamoun and Qiuhong Ke},
  booktitle = {NeurIPS},
  year   = {2025}
}
```

## License

MIT, see [LICENSE](LICENSE).
