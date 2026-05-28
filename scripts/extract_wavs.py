"""
Extract a subset of generated wavs from each eval's raw.npz for listening.
Picks the seed-0 100k checkpoint of every arch on Track A and dumps 24 samples each.
"""
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent

DIRS = {
    "lmlp":          REPO / "experiments/010_track_a_lmlp_s0/eval/step_0100000",
    "dit":           REPO / "experiments/011_track_a_dit_s0/eval/step_0100000",
    "lmlp_no_fnn_z": REPO / "experiments/012_track_a_lmlp_no_fnn_z_s0/eval/step_0100000",
    "unet":          REPO / "experiments/013_track_a_unet_s0/eval/step_0100000",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24,
                    help="how many samples per arch to dump")
    args = ap.parse_args()

    out_root = REPO / "results/listening_track_a_s0_step100k"
    out_root.mkdir(parents=True, exist_ok=True)

    for arch, eval_dir in DIRS.items():
        npz = eval_dir / "raw.npz"
        if not npz.exists():
            print(f"missing {npz}; skipping")
            continue
        data = np.load(npz)
        audio = data["sample_audio"].astype(np.float32)  # (N, 2, T)
        nn = data["nn_distances"]
        out_dir = out_root / arch
        out_dir.mkdir(exist_ok=True)
        n = min(args.n, audio.shape[0])
        for i in range(n):
            wav = audio[i].T  # (T, 2)
            peak = float(np.max(np.abs(wav)))
            if peak > 1e-8:
                wav = wav * (0.95 / peak)
            # filename embeds the NN-STFT-L1 score for that sample so you can
            # tell which were "best" vs "worst" by spectral-NN metric
            score = float(nn[i]) if i < len(nn) else 0.0
            sf.write(out_dir / f"sample_{i:03d}_stft{score:.3f}.wav",
                     wav, 44100, subtype="PCM_16")
        print(f"{arch}: wrote {n} wavs -> {out_dir}")

    # Also dump 16 real held-out test clips for "ground truth" reference
    cache = REPO / "data/test_audio_cache/track_a_test_n200.pt"
    if cache.exists():
        import torch
        d = torch.load(cache, weights_only=False, map_location="cpu")
        ref_dir = out_root / "_real_test_clips"
        ref_dir.mkdir(exist_ok=True)
        for i in range(min(16, len(d["audio"]))):
            a = d["audio"][i]  # (2, T)
            wav = a.T
            peak = float(np.max(np.abs(wav)))
            if peak > 1e-8:
                wav = wav * (0.95 / peak)
            sf.write(ref_dir / f"real_{i:03d}.wav", wav, 44100, subtype="PCM_16")
        print(f"real: wrote 16 reference wavs -> {ref_dir}")


if __name__ == "__main__":
    main()
