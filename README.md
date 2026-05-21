# RoadHeightFormer

Monocular road-surface height reconstruction in bird's-eye view (BEV) for preview-suspension applications.

## Overview

RoadHeightFormer is a transformer-based architecture that predicts dense BEV height maps of the road surface from a **single forward-facing camera image**. The model lifts image features into a metric BEV grid in front of the vehicle and regresses per-cell height, providing the kind of vertical profile that active suspension systems need to anticipate bumps, potholes and surface irregularities.

## Highlights

- **Monocular only** — no stereo rig, LiDAR or radar required at inference.
- **DINOv2 visual backbone** — leverages a self-supervised foundation model for strong, transferable features; both frozen and trainable variants are supported.
- **Composite supervision** — combines pixel-wise (MSE / L1), gradient, normal and structural losses to balance near-field accuracy against far-field stability.
- **Evaluated on two public datasets** — CARD and RSRD, using the standard RoadBEV protocol (absolute height error, RMSE).
- **Outperforms the RoadBEV stereo baseline** on both CARD and RSRD, and also surpasses DA3 in our evaluation.

## Repository status

> ⚠️ This README is provisional. A detailed description of the architecture, training procedure, configurations and reproduction steps will follow.

## Quick map of the repo

- `models/` — backbone, BEV lifting and prediction-head modules
- `cardset/`, `utils/dataset.py` — dataset loaders for CARD and RSRD
- `configs/` — YAML configurations for each ablation (frozen/trainable backbone, single/composite loss, full/cropped BEV range, etc.)
- `train.py`, `test.py` — training and evaluation entry points

## License

See [LICENSE](LICENSE).
