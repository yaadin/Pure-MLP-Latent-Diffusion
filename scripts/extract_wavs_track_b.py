"""
Extract a subset of Track B (8s loop) wavs from each arch's seed-0 step-100k
eval for listening.
"""
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parent.parent

DIRS = {
    "lmlp":          REPO / "experiments/020_track_b_lmlp_s0/eval/step_0100000",
    "dit":           REPO / "experiments/021_track_b_dit_s0/eval/step_0100000",
    "lmlp_no_fnn_z": REPO / "experiments/022_track_b_lmlp_no_fnn_z_s0/eval/step_0100000",
    "unet":          REPO / "experiments/023_track_b_unet_s0/eval/step_0100000",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16,
                    help="how many samples per arch")
    args = ap.parse_args()

    out_root = REPO / "results/listening_track_b_s0_step100k"
    out_root.mkdir(parents=True, exist_ok=True)

    for arch, eval_dir in DIRS.items():
        npz = eval_dir / "raw.npz"
        if not npz.exists():
            print(f"missing {npz}; skipping")
            continue
        data = np.load(npz)
        audio = data["sample_audio"].astype(np.float32)
        nn = data["nn_distances"]
        out_dir = out_root / arch
        out_dir.mkdir(exist_ok=True)
        n = min(args.n, audio.shape[0])
        for i in range(n):
            wav = audio[i].T
            peak = float(np.max(np.abs(wav)))
            if peak > 1e-8:
                wav = wav * (0.95 / peak)
            score = float(nn[i]) if i < len(nn) else 0.0
            sf.write(out_dir / f"sample_{i:03d}_stft{score:.3f}.wav",
                     wav, 44100, subtype="PCM_16")
        print(f"{arch}: wrote {n} wavs -> {out_dir}")

    # Real held-out test clips
    cache = REPO / "data/test_audio_cache/track_b_test_n200.pt"
    if cache.exists():
        d = torch.load(cache, weights_only=False, map_location="cpu")
        ref_dir = out_root / "_real_test_clips"
        ref_dir.mkdir(exist_ok=True)
        for i in range(min(12, len(d["audio"]))):
            a = d["audio"][i]
            wav = a.T
            peak = float(np.max(np.abs(wav)))
            if peak > 1e-8:
                wav = wav * (0.95 / peak)
            sf.write(ref_dir / f"real_{i:03d}.wav", wav, 44100, subtype="PCM_16")
        print(f"real: wrote 12 reference wavs -> {ref_dir}")


if __name__ == "__main__":
    main()
