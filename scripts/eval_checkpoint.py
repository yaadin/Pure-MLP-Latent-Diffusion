"""
Run the full evaluation pipeline for a single trained checkpoint.

Steps:
  1. Load checkpoint, sample N latents via DDIM.
  2. Decode latents to wavs via SAO VAE.
  3. Decode held-out test-set latents the same way (if first time).
  4. Compute:
        - STFT distance: mean nearest-neighbor STFT-L1 from generated to test set.
        - Latent-space stats (sample mean/std vs training-set mean/std).
  5. Save results to <ckpt_dir>/../eval/step_NNNNNNN/.

CLAP-FAD is left as a follow-up step; the orchestrator writes intermediate
artifacts (audio + embeddings stubs) that a later FAD job can consume.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from diffusers import AutoencoderOobleck

from src.eval.sample import load_model_from_ckpt, sample_latents
from src.eval.decode import decode_latents
from src.eval.metrics import (
    nn_stft_distance,
    multires_stft_l1,
    load_wav_dir,
)

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"


def get_test_audio_for_track(cfg: dict, max_n: int) -> list[np.ndarray]:
    """Decode the held-out test latents into waveforms. Cached so we only do
    this once per (track, max_n) combo."""
    latents_file = Path(cfg["data"]["latents_file"])
    split_file = Path(cfg["data"]["split_file"])
    split_dataset = cfg["data"]["split_dataset"]

    track_name = "a" if "track_a" in latents_file.name else "b"
    cache_dir = DATA / "test_audio_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"track_{track_name}_test_n{max_n}.pt"

    if cache_file.exists():
        d = torch.load(cache_file, weights_only=False, map_location="cpu")
        return d["audio"]

    print(f"[test cache miss for track {track_name}] decoding held-out test set...")
    d = torch.load(REPO / latents_file, weights_only=True, map_location="cpu")
    with open(REPO / split_file) as f:
        splits = json.load(f)
    test_ids = set(splits[split_dataset]["test"])
    keep_idx = [i for i, cid in enumerate(d["ids"]) if cid in test_ids]
    if len(keep_idx) > max_n:
        rng = np.random.default_rng(0)
        keep_idx = sorted(rng.choice(keep_idx, size=max_n, replace=False).tolist())
    lats = d["latents"][keep_idx]
    print(f"  selected {len(keep_idx)} test latents from {len(d['ids'])}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderOobleck.from_pretrained(
        "stabilityai/stable-audio-open-1.0", subfolder="vae"
    ).to(device).to(torch.bfloat16).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    audio = decode_latents(lats, vae, device, batch=4)  # (N, 2, T)
    audio_list = [audio[i] for i in range(audio.shape[0])]
    torch.save({"audio": audio_list}, cache_file)
    print(f"  cached -> {cache_file}")
    return audio_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=200,
                    help="how many held-out test clips to compare against")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_audio", action="store_true",
                    help="also dump generated wavs to disk (debugging)")
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- 1. sample -----
    model, cfg, step = load_model_from_ckpt(args.ckpt, device)
    print(f"loaded ckpt @ step {step:,}  arch={cfg['model']['arch']}")
    print(f"sampling {args.n_samples} latents...")
    t1 = time.time()
    sample_lats = sample_latents(
        model, cfg, args.n_samples, device, seed=args.seed, batch=32
    )
    print(f"  sampled in {time.time() - t1:.1f}s  shape={tuple(sample_lats.shape)}")
    del model
    torch.cuda.empty_cache()

    # ----- 2. decode samples -----
    print("decoding samples through SAO VAE...")
    t1 = time.time()
    vae = AutoencoderOobleck.from_pretrained(
        "stabilityai/stable-audio-open-1.0", subfolder="vae"
    ).to(device).to(torch.bfloat16).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    sample_audio = decode_latents(sample_lats, vae, device, batch=8)  # (N, 2, T)
    print(f"  decoded in {time.time() - t1:.1f}s  shape={sample_audio.shape}")
    del vae
    torch.cuda.empty_cache()

    sample_audio_list = [sample_audio[i] for i in range(sample_audio.shape[0])]

    # ----- 3. held-out test audio (cached) -----
    test_audio_list = get_test_audio_for_track(cfg, args.n_test)

    # ----- 4. metrics -----
    print(f"computing NN STFT-L1 distance "
          f"({len(sample_audio_list)} samples vs {len(test_audio_list)} test)...")
    t1 = time.time()
    nn_res = nn_stft_distance(sample_audio_list, test_audio_list)
    print(f"  computed in {time.time() - t1:.1f}s")
    print(f"  NN-STFT-L1: mean={nn_res['mean']:.4f}  std={nn_res['std']:.4f}  "
          f"min={nn_res['min']:.4f}  max={nn_res['max']:.4f}")

    # Sample mean/std vs training-set mean/std (in latent space)
    sample_lat_np = sample_lats.float().numpy()
    sample_stats = dict(
        mean=float(sample_lat_np.mean()),
        std=float(sample_lat_np.std()),
        per_ch_std_mean=float(sample_lat_np.std(axis=(0, 1)).mean()),
        per_pos_std_mean=float(sample_lat_np.std(axis=(0, 2)).mean()),
    )
    print(f"  sample latent stats: {sample_stats}")

    # ----- 5. save -----
    ckpt_path = args.ckpt
    eval_root = ckpt_path.parent.parent / "eval" / f"step_{step:07d}"
    eval_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "ckpt": str(ckpt_path),
        "step": step,
        "n_samples": args.n_samples,
        "n_test": len(test_audio_list),
        "seed": args.seed,
        "nn_stft_l1": {
            "mean": nn_res["mean"],
            "std":  nn_res["std"],
            "min":  nn_res["min"],
            "max":  nn_res["max"],
        },
        "sample_latent_stats": sample_stats,
        "wall_clock_s": round(time.time() - t0, 1),
    }
    with open(eval_root / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    # save raw arrays for later FAD or per-sample analysis
    np.savez(
        eval_root / "raw.npz",
        sample_audio=sample_audio.astype(np.float16),
        sample_latents=sample_lats.numpy().astype(np.float16),
        nn_distances=np.asarray(nn_res["nn_distances"], dtype=np.float32),
    )
    if args.save_audio:
        audio_dir = eval_root / "samples_wav"
        audio_dir.mkdir(exist_ok=True)
        for i, a in enumerate(sample_audio_list[:64]):
            peak = float(np.max(np.abs(a)))
            if peak > 1e-8:
                a = a * (0.95 / peak)
            sf.write(audio_dir / f"sample_{i:04d}.wav", a.T, 44100, subtype="PCM_16")
        print(f"  saved 64 wav samples to {audio_dir}")

    print(f"summary -> {eval_root / 'metrics.json'}")
    print(f"total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
