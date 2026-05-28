"""
Re-score NN-STFT-L1 for every existing eval dir using the GPU implementation,
without re-sampling. Reads sample audio from raw.npz, reads held-out reference
audio from the test_audio_cache, updates metrics.json in-place.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.eval.metrics import nn_stft_distance

REPO = Path(__file__).resolve().parent.parent


def main():
    eval_dirs = []
    for exp in sorted((REPO / "experiments").iterdir()):
        if not exp.is_dir():
            continue
        eval_root = exp / "eval"
        if not eval_root.exists():
            continue
        for step_dir in sorted(eval_root.iterdir()):
            if (step_dir / "raw.npz").exists() and (step_dir / "metrics.json").exists():
                eval_dirs.append(step_dir)
    print(f"will re-score {len(eval_dirs)} eval dirs")

    # Cache reference audio per track
    ref_cache: dict[str, list[np.ndarray]] = {}

    def get_refs(track_letter: str) -> list[np.ndarray]:
        if track_letter in ref_cache:
            return ref_cache[track_letter]
        p = REPO / "data" / "test_audio_cache" / f"track_{track_letter}_test_n200.pt"
        d = torch.load(p, weights_only=False, map_location="cpu")
        ref_cache[track_letter] = d["audio"]
        return d["audio"]

    t0 = time.time()
    for i, ed in enumerate(eval_dirs):
        exp_name = ed.parent.parent.name
        track_letter = "a" if "_track_a_" in exp_name else "b"
        try:
            raw = np.load(ed / "raw.npz")
            sample_audio = raw["sample_audio"].astype(np.float32)
            refs = get_refs(track_letter)
            sample_list = [sample_audio[k] for k in range(sample_audio.shape[0])]

            t1 = time.time()
            nn = nn_stft_distance(sample_list, refs, use_gpu=True)
            dt = time.time() - t1

            mfile = ed / "metrics.json"
            with open(mfile) as f:
                m = json.load(f)
            m["nn_stft_l1"] = {
                "mean": nn["mean"], "std": nn["std"],
                "min": nn["min"], "max": nn["max"],
            }
            m["nn_stft_l1_method"] = "gpu_torch_stft_v1"
            with open(mfile, "w") as f:
                json.dump(m, f, indent=2)
            # also overwrite the raw.npz nn_distances field
            np.savez(ed / "raw.npz",
                     sample_audio=raw["sample_audio"],
                     sample_latents=raw["sample_latents"],
                     nn_distances=np.asarray(nn["nn_distances"], dtype=np.float32))
            print(f"  [{i+1}/{len(eval_dirs)}] {exp_name}/{ed.name}  "
                  f"NN={nn['mean']:.4f}±{nn['std']:.4f}  ({dt:.1f}s)", flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(eval_dirs)}] {exp_name}/{ed.name}  FAIL: {e}", flush=True)

    print(f"\ntotal: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
