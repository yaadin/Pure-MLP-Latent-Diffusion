---
title: "Experiment 001: drums_pilot"
summary: "Go/no-go: does a small pure-MLP (L-MLP-style) denoiser produce structured drum samples through frozen SAO VAE on M4?"
status: planned
tags: [pilot, lmlp, drums, sao-vae, m4]
created: 2026-05-11
updated: 2026-05-11
---

## Hypothesis
A ~2-3M-param L-MLP-style denoiser, trained for ~100-200k steps on ~500-1000 CC-licensed drum loops (1-2s clips) encoded through the frozen SAO VAE, will:
1. Train without instability (smooth loss decrease, no NaNs).
2. Produce audibly-structured drum-like samples at end of training (not pure noise, not silence).
3. Reach STFT-distance lower than an untrained-model baseline.

If yes -> proceed to matched-param DiT baseline + 1-2 ablations -> Phase-1 report.
If no -> debug architecture / data pipeline / schedule before any cloud spend.

## Setup
- **Dataset:** ~500-1000 drum loops, CC0/CC-BY from Freesound or Kaggle. Curated for 1-2s clean kicks/snares/hats/loops. Cached as pre-encoded SAO VAE latents.
- **VAE:** Stable Audio Open VAE (`stabilityai/stable-audio-open-1.0`), frozen, downloaded from HuggingFace.
- **Model:** ULMLP, `src/models/lmlp.py`. Config: see `configs/lmlp_drums_pilot.yaml`. Embed dim 128, depth 6, mlp_ratio 2.0, merge_is_mlp=True. Expected param count: ~2-3M (verify after first forward).
- **Diffusion:** v-prediction, continuous-time cos/sin schedule (Salimans-Ho), uniform timestep sampling.
- **Baseline:** matched-param DiT-small (`src/models/dit_small.py`). NOT for this experiment; comes in 002.
- **Hyperparameters:** AdamW lr=2e-4, wd=0.03, betas=(0.9,0.9), warmup 1000 steps, batch 16, 100k steps. Mirrors L-MLP paper (Table 5) where possible.
- **Compute:** Apple M4, PyTorch MPS backend. Expected wall-clock: 2-4 days.
- **Seed(s):** 0 (will extend to {0,1,2} if time permits for variance).

## Run log
### 2026-05-11
- Experiment folder + repo skeleton created.
- **Implemented:** `src/vae/sao.py` (diffusers `AutoencoderOobleck.from_pretrained("stabilityai/stable-audio-open-1.0", subfolder="vae")`, channel-last wrapper, frozen), `src/data/augment.py` (pitch shift / gain / polarity / time shift for drum one-shots), `src/data/drums.py` (cached-latent Dataset), `scripts/encode_drums.py` (one-time encode pipeline).
- **Dataset choice:** Kaggle "drum-kit-sound-samples" (anubhavchhabra), 160 one-shot drum samples × 2s. Augmented to ~1280 latents via 8 variants per source. **License of Kaggle dataset to verify before publishing.**
- **VAE loading verified via HF docs:** `AutoencoderOobleck` is the official SAO VAE class in diffusers; loading from the SAO HF repo's `vae/` subfolder gives the same weights without pulling the 1B DiT or T5. License: Stability AI Community License (research / non-commercial).
- **Implemented:** `src/models/dit_small.py` (matched-param transformer baseline, learned pos embed, prepend timestep token), `src/diffusion/sampler.py` (deterministic DDIM in cos/sin parameter space for v-prediction), `src/train.py` (config-driven training loop with `--smoke` mode for synthetic-data sanity check).
- Pipeline now end-to-end runnable. Next: smoke-test on M4 via `python -m src.train --config configs/lmlp_drums_pilot.yaml --smoke` to confirm forward/backward + loss decrease + no NaNs before downloading the Kaggle dataset.

## Results
TBD.

## Conclusion
TBD.

## Next
- 002: matched-param DiT-small baseline, same data + schedule + sampler.
- 003: FNN_Z ablation (merge_is_mlp=false vs true).
- 004: maybe depth/width sweep if compute allows.
