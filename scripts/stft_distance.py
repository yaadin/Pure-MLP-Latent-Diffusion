from __future__ import annotations
import argparse
import datetime
import json
import statistics
from pathlib import Path
import soundfile as sf
import torch
import torchaudio.functional as AF
RESULTS_PATH = Path('paper/results/stft_distance.jsonl')
SAMPLE_RATE = 44100
CLIP_SECONDS = 2.0
FFT_SIZES = (512, 1024, 2048)
HOP_RATIOS = (4, 4, 4)

def _load_mono(path: Path) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype='float32', always_2d=True)
    if sr != SAMPLE_RATE:
        w = torch.from_numpy(data.T.copy())
        w = AF.resample(w, sr, SAMPLE_RATE)
        data = w.numpy().T
    wav = torch.from_numpy(data.mean(axis=1))
    target = int(CLIP_SECONDS * SAMPLE_RATE)
    if wav.shape[0] >= target:
        wav = wav[:target]
    else:
        wav = torch.cat([wav, torch.zeros(target - wav.shape[0])])
    return wav

def _stack_load(paths: list[Path]) -> torch.Tensor:
    return torch.stack([_load_mono(p) for p in paths])

def _multi_res_stft_mag(wavs: torch.Tensor) -> list[torch.Tensor]:
    out = []
    for n_fft, hop_ratio in zip(FFT_SIZES, HOP_RATIOS):
        hop = n_fft // hop_ratio
        spec = torch.stft(wavs, n_fft=n_fft, hop_length=hop, window=torch.hann_window(n_fft), return_complex=True, center=True).abs()
        out.append(spec)
    return out

def _pairwise_l1(gen_specs: list[torch.Tensor], ref_specs: list[torch.Tensor]) -> torch.Tensor:
    n_gen = gen_specs[0].shape[0]
    n_ref = ref_specs[0].shape[0]
    dist = torch.zeros(n_gen, n_ref)
    for gs, rs in zip(gen_specs, ref_specs):
        d = (gs.unsqueeze(1) - rs.unsqueeze(0)).abs().mean(dim=(-2, -1))
        dist = dist + d / len(gen_specs)
    return dist

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gen_dir', type=Path, required=True)
    p.add_argument('--gen_glob', type=str, default='*.wav')
    p.add_argument('--ref_dir', type=Path, default=Path('data/drums_raw'))
    p.add_argument('--ref_glob', type=str, default='**/*.wav')
    p.add_argument('--max_ref', type=int, default=160, help='Cap reference set size for speed.')
    args = p.parse_args()
    gen_files = sorted(args.gen_dir.glob(args.gen_glob))
    ref_files = sorted(args.ref_dir.glob(args.ref_glob))[:args.max_ref]
    if not gen_files:
        raise SystemExit(f'No generated files at {args.gen_dir}/{args.gen_glob}')
    if not ref_files:
        raise SystemExit(f'No reference files at {args.ref_dir}/{args.ref_glob}')
    print(f'loading {len(gen_files)} generated + {len(ref_files)} reference clips')
    gens = _stack_load(gen_files)
    refs = _stack_load(ref_files)
    print(f'computing multi-resolution STFTs (FFT sizes {FFT_SIZES})')
    gen_specs = _multi_res_stft_mag(gens)
    ref_specs = _multi_res_stft_mag(refs)
    print('computing pairwise distances')
    dist = _pairwise_l1(gen_specs, ref_specs)
    nn_dist = dist.min(dim=1).values
    nn_list = nn_dist.tolist()
    record = {'source': f'{args.gen_dir}/{args.gen_glob}', 'n_gen': len(gen_files), 'n_ref': len(ref_files), 'mean': sum(nn_list) / len(nn_list), 'median': statistics.median(nn_list), 'std': statistics.stdev(nn_list) if len(nn_list) > 1 else 0.0, 'min': min(nn_list), 'max': max(nn_list), 'per_sample_nn_dist': nn_list, 'timestamp': datetime.datetime.now().isoformat(timespec='seconds')}
    print()
    print(f'--- multi-res STFT distance to nearest training neighbor ---')
    print(f"source:   {record['source']}")
    print(f"n_gen:    {record['n_gen']}")
    print(f"n_ref:    {record['n_ref']}")
    print(f"mean:     {record['mean']:.4f}")
    print(f"median:   {record['median']:.4f}")
    print(f"std:      {record['std']:.4f}")
    print(f"min:      {record['min']:.4f}")
    print(f"max:      {record['max']:.4f}")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')
    print(f'\nappended to {RESULTS_PATH}')
if __name__ == '__main__':
    main()