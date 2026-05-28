"""
Compute bootstrap 95% confidence intervals on CLAP-FAD for every existing
Track B (and Track A) eval. Recomputes CLAP embeddings once per eval,
then resamples generated/reference indices with replacement B times and
recomputes Frechet distance.

Writes:
    experiments/*/eval/step_*/clap_fad.json  (adds 'ci95_lo', 'ci95_hi', 'boot_n')
"""
from __future__ import annotations
import argparse
import json
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
from scripts.run_clap_fad_batch import audio_to_48k_mono_batch, embed_audio

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["a", "b", "both"], default="both")
    ap.add_argument("--boot_n", type=int, default=1000)
    args = ap.parse_args()

    eval_dirs = []
    for exp in sorted((REPO / "experiments").iterdir()):
        if not exp.is_dir(): continue
        if "_track_a_" in exp.name and args.track not in ("a", "both"): continue
        if "_track_b_" in exp.name and args.track not in ("b", "both"): continue
        for step_dir in sorted((exp / "eval").iterdir() if (exp / "eval").exists() else []):
            if not (step_dir / "raw.npz").exists(): continue
            if not (step_dir / "clap_fad.json").exists(): continue
            eval_dirs.append(step_dir)
    print(f"will bootstrap {len(eval_dirs)} eval dirs (B={args.boot_n})", flush=True)

    clap = load_clap_model()
    print("CLAP loaded", flush=True)

    ref_emb_cache: dict[str, np.ndarray] = {}

    def get_refs(track_letter: str) -> np.ndarray:
        if track_letter in ref_emb_cache:
            return ref_emb_cache[track_letter]
        p = REPO / "data" / "test_audio_cache" / f"track_{track_letter}_test_n200.pt"
        d = torch.load(p, weights_only=False, map_location="cpu")
        audio_list = d["audio"]
        T = audio_list[0].shape[1]
        stacked = np.stack([
            a if a.shape[1] == T else np.pad(a, ((0, 0), (0, T - a.shape[1])))
            for a in audio_list
        ], axis=0)
        m48 = audio_to_48k_mono_batch(stacked, sr_in=44100)
        emb = embed_audio(clap, m48)
        ref_emb_cache[track_letter] = emb
        return emb

    rng = np.random.default_rng(0)
    t0 = time.time()
    for i, ed in enumerate(eval_dirs):
        track_letter = "a" if "_track_a_" in ed.parent.parent.name else "b"
        try:
            raw = np.load(ed / "raw.npz")
            sample_audio = raw["sample_audio"].astype(np.float32)
            ref_emb = get_refs(track_letter)

            # Embed generated samples
            m48 = audio_to_48k_mono_batch(sample_audio, sr_in=44100)
            sample_emb = embed_audio(clap, m48)

            # Point estimate
            mu_s, sig_s = gaussian_stats(sample_emb)
            mu_r, sig_r = gaussian_stats(ref_emb)
            fad_point = frechet_distance(mu_s, sig_s, mu_r, sig_r)

            # Bootstrap (resample both generated and reference with replacement)
            n_s, n_r = sample_emb.shape[0], ref_emb.shape[0]
            boots = np.empty(args.boot_n, dtype=np.float64)
            for b in range(args.boot_n):
                s_idx = rng.integers(0, n_s, size=n_s)
                r_idx = rng.integers(0, n_r, size=n_r)
                bs = sample_emb[s_idx]
                br = ref_emb[r_idx]
                mu_bs, sig_bs = gaussian_stats(bs)
                mu_br, sig_br = gaussian_stats(br)
                boots[b] = frechet_distance(mu_bs, sig_bs, mu_br, sig_br)
            lo = float(np.quantile(boots, 0.025))
            hi = float(np.quantile(boots, 0.975))

            # Merge into clap_fad.json
            ff = ed / "clap_fad.json"
            cf = json.load(open(ff))
            cf["clap_fad"] = float(fad_point)
            cf["ci95_lo"]  = lo
            cf["ci95_hi"]  = hi
            cf["boot_n"]   = args.boot_n
            with open(ff, "w") as f:
                json.dump(cf, f, indent=2)
            print(f"  [{i+1}/{len(eval_dirs)}] {ed.parent.parent.name}/{ed.name}  "
                  f"FAD={fad_point:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]", flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(eval_dirs)}] FAIL: {e}", flush=True)

    print(f"\ntotal: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
