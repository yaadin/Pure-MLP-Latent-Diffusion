from __future__ import annotations
import argparse
import math
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
ROWS = [('real drums', 'data/drums_raw', '**/*.wav'), ('unit-N latent decode (no model)', 'experiments/004_vae_prior_baseline/unit_normal', '*.wav'), ('dataset-matched random decode', 'experiments/004_vae_prior_baseline/dataset_matched', '*.wav'), ('001 L-MLP (FNN_Z=MLP), step 100k', 'experiments/001_drums_pilot/samples_wav', 'step_0100000_sample_*.wav'), ('002 DiT-small, step 100k', 'experiments/002_drums_dit_baseline/samples_wav', 'step_0100000_sample_*.wav'), ('003 L-MLP (FNN_Z=linear), step 100k', 'experiments/003_drums_lmlp_no_fnn_z/samples_wav', 'step_0100000_sample_*.wav')]
OUT_PATH = Path('paper/figures/spectrogram_grid.png')
SAMPLE_RATE = 44100
N_MELS = 80
N_FFT = 2048
HOP = 512

def _load_mono(path: Path) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype='float32', always_2d=True)
    if sr != SAMPLE_RATE:
        import torchaudio.functional as AF
        w = torch.from_numpy(data.T.copy())
        w = AF.resample(w, sr, SAMPLE_RATE)
        data = w.numpy().T
    return torch.from_numpy(data.mean(axis=1))

def _melspec_db(wav: torch.Tensor) -> np.ndarray:
    mel = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0, normalized=False, center=True)(wav.unsqueeze(0))
    db = T.AmplitudeToDB(top_db=80.0)(mel)
    return db.squeeze(0).numpy()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_cols', type=int, default=4, help='samples per row')
    p.add_argument('--clip_seconds', type=float, default=2.0)
    args = p.parse_args()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(ROWS)
    fig, axes = plt.subplots(n_rows, args.n_cols, figsize=(2.4 * args.n_cols, 1.6 * n_rows), dpi=150, squeeze=False)
    rng = random.Random(0)
    for r, (label, src, glob) in enumerate(ROWS):
        files = sorted(Path(src).glob(glob))
        if not files:
            print(f"  WARN: no files for row '{label}' at {src}/{glob}")
            for c in range(args.n_cols):
                axes[r, c].axis('off')
            continue
        if label == 'real drums':
            by_subdir: dict[Path, list[Path]] = {}
            for f in files:
                rel = f.relative_to(Path(src))
                key = rel.parts[0] if len(rel.parts) > 1 else Path('.')
                by_subdir.setdefault(key, []).append(f)
            picked = []
            for sub in sorted(by_subdir.keys())[:args.n_cols]:
                picked.append(rng.choice(by_subdir[sub]))
            while len(picked) < args.n_cols and files:
                picked.append(rng.choice(files))
            files = picked
        for c in range(args.n_cols):
            ax = axes[r, c]
            if c >= len(files):
                ax.axis('off')
                continue
            wav = _load_mono(files[c])
            target = int(args.clip_seconds * SAMPLE_RATE)
            if wav.shape[0] >= target:
                wav = wav[:target]
            else:
                pad = torch.zeros(target - wav.shape[0])
                wav = torch.cat([wav, pad])
            db = _melspec_db(wav)
            ax.imshow(db, aspect='auto', origin='lower', cmap='magma', vmin=-80, vmax=0)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=7, rotation=0, ha='right', va='center')
    fig.suptitle('Mel-spectrograms (80 mels, 2 s clips): real data, VAE-prior controls, and three trained denoisers at step 100k', fontsize=10)
    fig.tight_layout(rect=(0.02, 0.0, 1.0, 0.97))
    fig.savefig(OUT_PATH)
    print(f'wrote {OUT_PATH}')
if __name__ == '__main__':
    main()