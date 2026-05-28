from __future__ import annotations
import argparse
from pathlib import Path
import soundfile as sf
import torch
import yaml
from src.diffusion import ddim_sample
from src.models import build_model
from src.vae.sao import SAOVAE
DEFAULT_RUNS = [('001_lmlp', 'configs/lmlp_drums_pilot.yaml', 'experiments/001_drums_pilot/checkpoints/step_0100000.pt'), ('002_dit', 'configs/dit_drums_pilot.yaml', 'experiments/002_drums_dit_baseline/checkpoints/step_0100000.pt')]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    p.add_argument('--num_samples', type=int, default=4, help='Number of samples per seed (drawn from a single batch).')
    p.add_argument('--num_steps', type=int, default=50, help='DDIM sampler step count.')
    p.add_argument('--device', type=str, default='mps')
    p.add_argument('--out_root', type=Path, default=Path('experiments/multi_seed_compare'))
    args = p.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f'loading SAO VAE on device={device}')
    vae = SAOVAE(device=device)
    sample_rate = SAOVAE.SAMPLE_RATE
    for label, cfg_path, ckpt_path in DEFAULT_RUNS:
        if not Path(ckpt_path).exists():
            print(f'skip {label}: checkpoint not found at {ckpt_path}')
            continue
        print(f'--- {label} ---')
        cfg = yaml.safe_load(open(cfg_path))
        mcfg = dict(cfg['model'])
        mcfg['seq_len'] = cfg['data']['seq_len']
        model = build_model(mcfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        model.eval()
        run_dir = args.out_root / label
        run_dir.mkdir(parents=True, exist_ok=True)
        for s in args.seeds:
            z = ddim_sample(model, shape=(args.num_samples, cfg['data']['seq_len'], cfg['model']['latent_channels']), num_steps=args.num_steps, device=device, seed=s)
            wav = vae.decode(z).cpu().clamp(-1.0, 1.0)
            for i in range(wav.shape[0]):
                w = wav[i]
                peak = w.abs().max().item()
                if peak > 0:
                    w = w * (0.95 / peak)
                out_path = run_dir / f'seed_{s:02d}_sample_{i:02d}.wav'
                sf.write(str(out_path), w.transpose(0, 1).numpy(), sample_rate, subtype='PCM_16')
            print(f'  seed={s}: wrote {args.num_samples} .wav')
        del model
        if device.type == 'mps':
            torch.mps.empty_cache()
    print(f'done -> {args.out_root}')
if __name__ == '__main__':
    main()