from __future__ import annotations
import argparse
from pathlib import Path
import soundfile as sf
import torch
from src.vae.sao import SAOVAE

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples_dir', type=Path, required=True)
    p.add_argument('--out_dir', type=Path, required=True)
    p.add_argument('--device', type=str, default='mps')
    p.add_argument('--peak_normalize', action='store_true', help='Scale each .wav so peak = 0.95 (avoids clipping when written as int16-able audio).')
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    latent_files = sorted(args.samples_dir.glob('latents_step_*.pt'))
    if not latent_files:
        raise SystemExit(f'No latents_step_*.pt files found in {args.samples_dir}.')
    print(f'loading SAO VAE on device={args.device}')
    vae = SAOVAE(device=args.device)
    sample_rate = SAOVAE.SAMPLE_RATE
    for lf in latent_files:
        z = torch.load(lf, map_location='cpu', weights_only=True)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        assert z.dim() == 3 and z.shape[-1] == SAOVAE.LATENT_CHANNELS, f'unexpected latent shape in {lf.name}: {tuple(z.shape)}'
        z = z.to(args.device)
        wav = vae.decode(z).cpu()
        wav = wav.clamp(-1.0, 1.0)
        step_str = lf.stem.split('_')[-1]
        for i in range(wav.shape[0]):
            w = wav[i]
            if args.peak_normalize:
                peak = w.abs().max().item()
                if peak > 0:
                    w = w * (0.95 / peak)
            out_path = args.out_dir / f'step_{step_str}_sample_{i:02d}.wav'
            sf.write(str(out_path), w.transpose(0, 1).numpy(), sample_rate, subtype='PCM_16')
        print(f'decoded {lf.name} -> {wav.shape[0]} .wav files')
    print(f'done -> {args.out_dir}')
if __name__ == '__main__':
    main()