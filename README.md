# Pure-MLP Latent Diffusion for Drum Sound Generation

Code release for the paper *Beyond Attention and Convolution: Pure-MLP Latent
Diffusion for Drum Sound Generation*.

A strictly attention-free and convolution-free MLP denoiser (the L-MLP block of
[Hu and Rostami, 2024](https://arxiv.org/abs/2406.01853), adapted to audio)
trained in the frozen latent space of [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0).
The paper compares L-MLP against matched-parameter DiT-small and 1D U-Net
baselines on two drum tasks: FSD50K one-shots and GMD drum loops.

## Repository layout

```
src/
  data/          dataset + latent caching
  diffusion/     v-prediction schedule, DDIM sampler
  eval/          NN-STFT-L1 + CLAP-FAD metrics
  models/        L-MLP, DiT-small, 1D U-Net denoisers
  vae/           thin wrapper around the SAO AutoencoderOobleck
  train.py       training entry point
scripts/         data prep, evaluation, figure generation, batch runners
configs/         one YAML per trained model (38 total)
```

The paper LaTeX source is not bundled with the code release. Bulk data
under `data/`, training-time artifacts under `experiments/NNN_<name>/`
(checkpoints, generated samples, raw eval audio), and local logs under
`logs/` are also not tracked.

## Setup

Python 3.12 with CUDA 12.8 and PyTorch 2.11. CPU-only inference also works.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

LAION-CLAP weights (~1 GB) are downloaded on first CLAP-FAD evaluation.

## Reproducing the paper

Phase 0: pull the datasets.

* FSD50K from [Zenodo](https://zenodo.org/records/4060432), filtered to drum
  labels (see `scripts/encode_drums.py`).
* GMD from [Magenta](https://magenta.tensorflow.org/datasets/groove),
  `2.0.0-full`.

Phase 1: cache latents through the SAO VAE.

```bash
python scripts/encode_drums.py --track a
python scripts/encode_drums.py --track b
```

Phase 2: train the 38-model matrix.

```bash
# Main 24-model matrix (4 archs x 2 tracks x 3 seeds at 100k steps)
bash scripts/run_matrix.sh

# Extra Track B seeds (s3, s4 for all 4 archs at 100k)
# Plus DiT-small extended to 200k x 3 seeds
bash scripts/run_critique_overnight.sh

# L-MLP-main extended to 200k x 3 seeds (matched-budget comparison)
bash scripts/run_lmlp_200k_overnight.sh
```

Phase 3: evaluate, aggregate, and render figures.

```bash
python scripts/run_clap_fad_batch.py --track both
python scripts/aggregate_results.py
python scripts/make_figures.py
```

The aggregate writes `results/phase2_summary.md` (markdown tables) and
`results/phase2_table.tsv` (raw rows). The figure script writes to
`paper/figures/` (created locally; not tracked in this repository).

Single-checkpoint evaluation:

```bash
python scripts/eval_checkpoint.py \
  --ckpt experiments/020_track_b_lmlp_s0/checkpoints/step_0100000.pt \
  --n_samples 500 --n_test 200
```

## Hardware

A single NVIDIA RTX 5070 (12 GB) was used for all training. Per-run times:
~46-80 min for the 100k Track A/B runs, ~76 min for the 200k Track B runs.
The full 38-model pipeline (training + evaluation + figures) takes
approximately 30 hours of GPU time.

## Citation

Citation pending. The paper is in submission; this section will be updated
once it appears on arXiv or in a venue.

## License

Code is released under the MIT License (see `LICENSE`). The two datasets used
for training and evaluation are distributed by their original authors under CC
licenses:

* FSD50K: CC BY 4.0 / CC0 / sampling+ (per-clip, see metadata)
* Groove MIDI Dataset (GMD): CC BY 4.0
