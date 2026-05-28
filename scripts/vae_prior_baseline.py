from __future__ import annotations
import argparse
import random
from pathlib import Path
import soundfile as sf
import torch
from src.vae.sao import SAOVAE

def _dataset_stats(latents_dir: Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    files = sorted(latents_dir.glob('*.pt'))
    if not files:
        raise SystemExit(f'No latents in {latents_dir}')
    stacked = torch.stack([torch.load(f, map_location='cpu', weights_only=True) for f in files])
    return (stacked.mean(dim=0), stacked.std(dim=0), stacked.shape[1])

def _save_wavs(wav: torch.Tensor, sample_rate: int, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = wav.clamp(-1.0, 1.0)
    for i in range(wav.shape[0]):
        w = wav[i]
        peak = w.abs().max().item()
        if peak > 0:
            w = w * (0.95 / peak)
        sf.write(str(out_dir / f'{prefix}_{i:02d}.wav'), w.transpose(0, 1).numpy(), sample_rate, subtype='PCM_16')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--latents_dir', type=Path, default=Path('data/drums_pilot_latents'))
    p.add_argument('--out_dir', type=Path, default=Path('experiments/004_vae_prior_baseline'))
    p.add_argument('--num_samples', type=int, default=8)
    p.add_argument('--device', type=str, default='mps')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    print('computing dataset latent statistics...')
    mu, sigma, seq_len = _dataset_stats(args.latents_dir)
    print(f'  seq_len={seq_len}  mean(|mu|)={mu.abs().mean():.3f}  mean(sigma)={sigma.mean():.3f}')
    print(f'loading SAO VAE on device={args.device}')
    vae = SAOVAE(device=args.device)
    sample_rate = SAOVAE.SAMPLE_RATE
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    shape = (args.num_samples, seq_len, SAOVAE.LATENT_CHANNELS)
    z_unit = torch.randn(*shape)
    wav_unit = vae.decode(z_unit.to(args.device)).cpu()
    _save_wavs(wav_unit, sample_rate, args.out_dir / 'unit_normal', 'unit_normal')
    z_matched = torch.randn(*shape) * sigma + mu
    wav_matched = vae.decode(z_matched.to(args.device)).cpu()
    _save_wavs(wav_matched, sample_rate, args.out_dir / 'dataset_matched', 'dataset_matched')
    all_files = list(args.latents_dir.glob('*.pt'))
    picked = random.sample(all_files, args.num_samples)
    z_real = torch.stack([torch.load(f, map_location='cpu', weights_only=True) for f in picked])
    wav_real = vae.decode(z_real.to(args.device)).cpu()
    _save_wavs(wav_real, sample_rate, args.out_dir / 'dataset_resample', 'dataset_resample')
    print(f'done -> {args.out_dir}/{{unit_normal,dataset_matched,dataset_resample}}/')
    print('\nListen pair-wise:')
    print('  unit_normal      vs trained-model-step-100k samples')
    print('  dataset_matched  vs trained-model-step-100k samples')
    print('  dataset_resample vs trained-model-step-100k samples')
if __name__ == '__main__':
    main()