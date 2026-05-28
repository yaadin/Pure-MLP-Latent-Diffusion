---
title: "Experiment 003: drums_lmlp_no_fnn_z"
summary: "L-MLP without the FNN_Z merging projection. Ablates the load-bearing design choice from Hu & Rostami (NeurIPS 2024)."
status: planned
tags: [pilot, lmlp, ablation, drums, m4]
created: 2026-05-11
updated: 2026-05-11
---

## Hypothesis
Per Hu & Rostami (NeurIPS 2024) Table 1, replacing the MLP merging projection (FNN_Z)
with a plain `Add + None` collapses image FID > 100. Does the same load-bearing
property hold for short audio latent sequences?

Two outcomes both interesting:
- **Same collapse:** FNN_Z is universal across modalities; replicates a key finding
  from the image domain in audio.
- **No collapse (or much less severe):** FNN_Z's role is less critical at short
  sequence lengths or for audio latents; novel observation.

## Setup
- **Dataset:** `data/drums_pilot_latents/` (1280 cached latents, same as 001 & 002).
- **Model:** ULMLP with `merge_is_mlp=false`, a linear projection in place of the MLP merger.
  Embed_dim=128, depth=6, mlp_ratio=2.0. Param count slightly lower than 001 because
  FNN_Z drops from 2-layer MLP to 1-layer linear. (Will print at run start.)
- **Diffusion:** v-prediction, cos/sin schedule; identical to 001 & 002.
- **Hyperparameters:** AdamW lr=2e-4 wd=0.03 betas=(0.9,0.9), warmup 1000, batch 16, 100k steps.
- **Sampling:** DDIM 50 steps, 8 samples every 5k steps.
- **Compute:** M4 MPS, ~30 min expected.
- **Seed:** 0.

## Run log
### 2026-05-11
- Config created at `configs/lmlp_no_fnn_z_drums.yaml`.
- Pending: launch after 002 decoding.

## Results
TBD.

## Conclusion
TBD.
