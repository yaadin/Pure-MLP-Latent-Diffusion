"""
Compute CLAP-FAD for all completed evaluations.

Sources:
    experiments/*/eval/step_*/raw.npz   (sample audio, fp16, shape (N, 2, T))
    data/test_audio_cache/track_{a,b}_test_n200.pt  (held-out reference audio)

Output:
    experiments/*/eval/step_*/clap_fad.json
    (skipped if file already exists)

Loads CLAP once and reuses across all eval dirs.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.eval.metrics import frechet_distance, gaussian_stats
from src.eval.clap_fad import (
    CLAP_SAMPLE_RATE,
    CLAP_TARGET_SECONDS,
    _resample_to_48k_mono,
    load_clap_model,
)

REPO = Path(__file__).resolve().parent.parent
TARGET_SAMPLES = int(CLAP_TARGET_SECONDS * CLAP_SAMPLE_RATE)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def audio_to_48k_mono_batch(audio_2ch: np.ndarray, sr_in: int) -> np.ndarray:
    """audio_2ch: (N, 2, T) -> (N, TARGET_SAMPLES) at 48kHz mono."""
    out = np.zeros((audio_2ch.shape[0], TARGET_SAMPLES), dtype=np.float32)
    for i in range(audio_2ch.shape[0]):
        m = _resample_to_48k_mono(audio_2ch[i].astype(np.float32), sr_in)
        if len(m) < TARGET_SAMPLES:
            out[i, :len(m)] = m
        else:
            out[i] = m[:TARGET_SAMPLES]
    return out


def embed_audio(clap_model, audio_48k_mono: np.ndarray,
                batch_size: int = 16) -> np.ndarray:
    """audio_48k_mono: (N, T) -> (N, 512)"""
    embs = []
    for i in range(0, audio_48k_mono.shape[0], batch_size):
        chunk = audio_48k_mono[i:i + batch_size]
        with torch.no_grad():
            x = torch.from_numpy(chunk).cuda() if torch.cuda.is_available() else torch.from_numpy(chunk)
            e = clap_model.get_audio_embedding_from_data(x=x, use_tensor=True)
        embs.append(e.cpu().numpy())
    return np.concatenate(embs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["a", "b", "both"], default="both",
                    help="which track to process")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if clap_fad.json exists")
    args = ap.parse_args()

    eval_dirs: list[Path] = []
    for exp in sorted((REPO / "experiments").iterdir()):
        if not exp.is_dir():
            continue
        # filter by track
        if "_track_a_" in exp.name and args.track not in ("a", "both"):
            continue
        if "_track_b_" in exp.name and args.track not in ("b", "both"):
            continue
        for step_dir in sorted((exp / "eval").iterdir() if (exp / "eval").exists() else []):
            raw_npz = step_dir / "raw.npz"
            fad_json = step_dir / "clap_fad.json"
            if not raw_npz.exists():
                continue
            if fad_json.exists() and not args.force:
                continue
            eval_dirs.append(step_dir)

    if not eval_dirs:
        log("nothing to do")
        return
    log(f"will process {len(eval_dirs)} eval dirs")
    for ed in eval_dirs[:10]:
        log(f"  - {ed.relative_to(REPO)}")
    if len(eval_dirs) > 10:
        log(f"  ... and {len(eval_dirs) - 10} more")

    log("loading CLAP...")
    clap = load_clap_model()
    log("  CLAP loaded")

    # Cache reference embeddings per track so we don't recompute.
    ref_emb_cache: dict[str, np.ndarray] = {}

    def get_ref_embeddings(track_letter: str) -> np.ndarray:
        if track_letter in ref_emb_cache:
            return ref_emb_cache[track_letter]
        cache_file = REPO / "data" / "test_audio_cache" / f"track_{track_letter}_test_n200.pt"
        if not cache_file.exists():
            raise RuntimeError(f"missing reference cache: {cache_file}")
        d = torch.load(cache_file, weights_only=False, map_location="cpu")
        audio_list = d["audio"]  # list of (2, T) float32
        log(f"  computing reference embeddings for track {track_letter}: "
            f"{len(audio_list)} clips")
        # Pad shortest to longest for stacking; all at the same SR (44100)
        stacked = np.stack([
            a if a.shape[1] == audio_list[0].shape[1]
            else np.pad(a, ((0, 0), (0, audio_list[0].shape[1] - a.shape[1])))
            for a in audio_list
        ], axis=0)
        m48 = audio_to_48k_mono_batch(stacked, sr_in=44100)
        emb = embed_audio(clap, m48)
        ref_emb_cache[track_letter] = emb
        log(f"    ref embeddings: {emb.shape}")
        return emb

    t0 = time.time()
    for i, ed in enumerate(eval_dirs):
        track_letter = "a" if "_track_a_" in ed.parent.parent.name else "b"
        try:
            data = np.load(ed / "raw.npz")
            sample_audio = data["sample_audio"].astype(np.float32)  # (N, 2, T)
        except Exception as e:
            log(f"  [{i+1}/{len(eval_dirs)}] {ed.relative_to(REPO)}: LOAD FAIL {e}")
            continue

        ref_emb = get_ref_embeddings(track_letter)

        log(f"  [{i+1}/{len(eval_dirs)}] {ed.relative_to(REPO)}  "
            f"samples={sample_audio.shape}")
        t1 = time.time()
        m48_samples = audio_to_48k_mono_batch(sample_audio, sr_in=44100)
        sample_emb = embed_audio(clap, m48_samples)
        mu_s, sig_s = gaussian_stats(sample_emb)
        mu_r, sig_r = gaussian_stats(ref_emb)
        fad = frechet_distance(mu_s, sig_s, mu_r, sig_r)
        with open(ed / "clap_fad.json", "w") as f:
            json.dump({
                "clap_fad": fad,
                "n_samples": int(sample_emb.shape[0]),
                "n_refs": int(ref_emb.shape[0]),
                "track": track_letter,
            }, f, indent=2)
        log(f"    CLAP-FAD = {fad:.4f}  ({time.time()-t1:.1f}s)")

    log(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
