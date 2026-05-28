---
title: "Experiment 002: drums_dit_baseline"
summary: "Matched-param DiT-small baseline against the L-MLP (001), same data + schedule + sampler. Only the denoiser block differs."
status: planned
tags: [pilot, dit, baseline, drums, m4]
created: 2026-05-11
updated: 2026-05-11
---

## Hypothesis
A 944k-param DiT-small denoiser (transformer block + learned positional embedding +
prepend timestep token), trained for the same 100k steps on the same drum latents
under the same v-prediction schedule, will produce drum-like audio samples and a
final loss within similar magnitude of the L-MLP (001).

Purpose: a fair architectural comparison. If L-MLP (001) reaches ~0.55 final loss and
produces recognizable drums, DiT (002) should be in the same ballpark (matched
params). Any large gap either way is the headline result.

## Setup
- **Dataset:** `data/drums_pilot_latents/`, same 1280 cached latents as 001.
- **Model:** DiTSmall, embed_dim=128, depth=6, num_heads=4, mlp_ratio=4.0. Params: 944,320.
- **Diffusion:** v-prediction, cos/sin schedule, uniform-t sampling; identical to 001.
- **Hyperparameters:** AdamW lr=2e-4 wd=0.03 betas=(0.9,0.9), warmup 1000, batch 16, 100k steps.
- **Sampling:** DDIM 50 steps, 8 samples every 5k training steps.
- **Compute:** M4 MPS, ~30-35 min expected (similar to 001).
- **Seed:** 0.

## Run log
### 2026-05-11
- Config created at `configs/dit_drums_pilot.yaml`.
- Pending: launch.

## Results
TBD.

## Conclusion
TBD.
