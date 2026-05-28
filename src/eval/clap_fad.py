"""
CLAP-based Frechet Audio Distance.

Embeds audio via LAION CLAP (audio branch) and computes the Frechet distance
between the empirical Gaussian fit of a set of generated samples and that of
a reference set. Lighter than the `frechet-audio-distance` package's tangled
torchhub VGGish path, and gives semantic-perceptual proximity.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch

from src.eval.metrics import frechet_distance, gaussian_stats


CLAP_SAMPLE_RATE = 48000   # LAION CLAP audio branch expects 48 kHz mono
CLAP_TARGET_SECONDS = 10.0


def _resample_to_48k_mono(audio: np.ndarray, sr_in: int) -> np.ndarray:
    """audio: (ch, T) or (T,) at sr_in -> mono 48kHz numpy array (T,)."""
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    if sr_in == CLAP_SAMPLE_RATE:
        return audio.astype(np.float32)
    # use scipy resample_poly for speed + simplicity
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr_in, CLAP_SAMPLE_RATE)
    up = CLAP_SAMPLE_RATE // g
    down = sr_in // g
    out = resample_poly(audio, up, down).astype(np.float32)
    return out


def load_clap_model():
    """Load the LAION CLAP model. Downloads checkpoint (~1GB) on first use."""
    import laion_clap
    print("loading LAION CLAP (HTSAT-tiny; matches default ckpt)...", flush=True)
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")
    # `load_ckpt()` with no args grabs the default LAION CLAP audio-only ckpt.
    # First call downloads from HF; cached afterwards.
    model.load_ckpt()
    model.eval()
    return model


def embed_audio_files(clap_model, wav_files: list[Path],
                      batch_size: int = 8) -> np.ndarray:
    """Return (N, 512) numpy array of CLAP audio embeddings."""
    embs: list[np.ndarray] = []
    target_samples = int(CLAP_TARGET_SECONDS * CLAP_SAMPLE_RATE)
    for i in range(0, len(wav_files), batch_size):
        batch = wav_files[i:i + batch_size]
        wavs = []
        for f in batch:
            a, sr = sf.read(str(f), dtype="float32", always_2d=True)
            a = a.T  # (ch, T)
            a48 = _resample_to_48k_mono(a, sr)
            # pad or trim to fixed 10s (CLAP's expected input)
            if len(a48) < target_samples:
                a48 = np.pad(a48, (0, target_samples - len(a48)))
            else:
                a48 = a48[:target_samples]
            wavs.append(a48)
        arr = np.stack(wavs, axis=0)
        with torch.no_grad():
            x = torch.from_numpy(arr).cuda() if torch.cuda.is_available() else torch.from_numpy(arr)
            emb = clap_model.get_audio_embedding_from_data(x=x, use_tensor=True)
        embs.append(emb.cpu().numpy())
    return np.concatenate(embs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dir", type=Path, required=True,
                    help="dir of generated .wav files")
    ap.add_argument("--reference_dir", type=Path, required=True,
                    help="dir of held-out reference .wav files")
    ap.add_argument("--out", type=Path, required=True,
                    help="json output path")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    sample_wavs = sorted(args.samples_dir.glob("*.wav"))
    ref_wavs = sorted(args.reference_dir.glob("*.wav"))
    if not sample_wavs or not ref_wavs:
        print(f"empty input: samples={len(sample_wavs)} refs={len(ref_wavs)}",
              file=sys.stderr)
        sys.exit(1)

    clap = load_clap_model()
    print(f"embedding {len(sample_wavs)} samples + {len(ref_wavs)} refs...")
    e_s = embed_audio_files(clap, sample_wavs, batch_size=args.batch_size)
    e_r = embed_audio_files(clap, ref_wavs,   batch_size=args.batch_size)
    print(f"  samples emb: {e_s.shape}  refs emb: {e_r.shape}")

    mu_s, sig_s = gaussian_stats(e_s)
    mu_r, sig_r = gaussian_stats(e_r)
    fad = frechet_distance(mu_s, sig_s, mu_r, sig_r)
    print(f"  CLAP-FAD: {fad:.4f}")

    import json
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "clap_fad": fad,
            "n_samples": len(sample_wavs),
            "n_refs": len(ref_wavs),
            "samples_dir": str(args.samples_dir),
            "reference_dir": str(args.reference_dir),
        }, f, indent=2)


if __name__ == "__main__":
    main()
