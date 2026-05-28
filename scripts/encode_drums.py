from __future__ import annotations
import argparse
import math
from pathlib import Path
import soundfile as sf
import torch
import torchaudio.functional as AF
from src.data.augment import augment_one_shot
from src.vae.sao import SAOVAE
SUPPORTED_EXTS = {'.wav', '.flac', '.ogg'}

def _load_stereo_44k(path: Path, target_sr: int=44100) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype='float32', always_2d=True)
    wav = torch.from_numpy(data.T.copy())
    if sr != target_sr:
        wav = AF.resample(wav, sr, target_sr)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    return wav

def _fix_length(wav: torch.Tensor, num_samples: int) -> torch.Tensor:
    T = wav.shape[1]
    if T >= num_samples:
        return wav[:, :num_samples]
    pad = torch.zeros(wav.shape[0], num_samples - T, dtype=wav.dtype)
    return torch.cat([wav, pad], dim=1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw_dir', type=Path, required=True)
    p.add_argument('--out_dir', type=Path, required=True)
    p.add_argument('--clip_seconds', type=float, default=2.0)
    p.add_argument('--variants_per_clip', type=int, default=8)
    p.add_argument('--device', type=str, default='mps', help='mps | cpu | cuda')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = SAOVAE.SAMPLE_RATE
    target_samples = int(args.clip_seconds * sample_rate)
    raw_files = sorted([f for f in args.raw_dir.rglob('*') if f.suffix.lower() in SUPPORTED_EXTS])
    if not raw_files:
        raise SystemExit(f'No audio files found in {args.raw_dir}')
    print(f'found {len(raw_files)} source files; will produce {len(raw_files) * args.variants_per_clip} latents')
    vae = SAOVAE(device=args.device)
    semitone_grid = [0.0, 0.0, -1.0, 1.0, -2.0, 2.0, -1.0, 1.0]
    if len(semitone_grid) < args.variants_per_clip:
        semitone_grid = (semitone_grid * math.ceil(args.variants_per_clip / len(semitone_grid)))[:args.variants_per_clip]
    else:
        semitone_grid = semitone_grid[:args.variants_per_clip]
    pending: list[tuple[Path, int, torch.Tensor]] = []
    for fi, src in enumerate(raw_files):
        wav = _load_stereo_44k(src, sample_rate)
        wav = _fix_length(wav, target_samples)
        for vi in range(args.variants_per_clip):
            aug_seed = args.seed * 10000 + fi * 100 + vi
            aug = augment_one_shot(wav, sample_rate=sample_rate, seed=aug_seed, pitch_shift_semitones=semitone_grid[vi])
            pending.append((src, vi, aug))
    device = torch.device(args.device)
    for start in range(0, len(pending), args.batch_size):
        chunk = pending[start:start + args.batch_size]
        batch = torch.stack([w for _, _, w in chunk], dim=0).to(device)
        latents = vae.encode(batch).cpu()
        for (src, vi, _), z in zip(chunk, latents):
            out_path = args.out_dir / f'{src.stem}_v{vi:02d}.pt'
            torch.save(z.contiguous(), out_path)
        print(f'encoded {min(start + args.batch_size, len(pending))} / {len(pending)}')
    print(f'done -> {args.out_dir}')
if __name__ == '__main__':
    main()