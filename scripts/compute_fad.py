from __future__ import annotations
import argparse
import datetime
import json
import shutil
import tempfile
from pathlib import Path
RESULTS_PATH = Path('paper/results/fad.jsonl')

def _gather(src: Path, pattern: str) -> list[Path]:
    return sorted(src.glob(pattern))

def _stage(files: list[Path], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, out / f.name)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gen_dir', type=Path, required=True)
    p.add_argument('--gen_glob', type=str, default='*.wav')
    p.add_argument('--ref_dir', type=Path, default=Path('data/drums_raw'))
    p.add_argument('--ref_glob', type=str, default='**/*.wav')
    p.add_argument('--model', type=str, default='pann', choices=['pann', 'vggish', 'clap'], help='Embedder. PANN (PyTorch-native) is recommended.')
    p.add_argument('--device', type=str, default='cpu', help='FAD package handles its own placement; CPU is reliable on macOS.')
    args = p.parse_args()
    import torch
    _original_load = torch.load

    def _legacy_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _original_load(*args, **kwargs)
    torch.load = _legacy_load
    try:
        from frechet_audio_distance import FrechetAudioDistance
    except ImportError as e:
        raise SystemExit(f'Missing dep: pip install frechet-audio-distance\n  underlying error: {e}')
    gen_files = _gather(args.gen_dir, args.gen_glob)
    ref_files = _gather(args.ref_dir, args.ref_glob)
    if not gen_files:
        raise SystemExit(f'No generated files at {args.gen_dir}/{args.gen_glob}')
    if not ref_files:
        raise SystemExit(f'No reference files at {args.ref_dir}/{args.ref_glob}')
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        gen_staged = td_path / 'gen'
        ref_staged = td_path / 'ref'
        _stage(gen_files, gen_staged)
        _stage(ref_files, ref_staged)
        sr_per_model = {'pann': 16000, 'vggish': 16000, 'clap': 48000}
        sr = sr_per_model.get(args.model, 16000)
        print(f'computing FAD ({args.model}, sr={sr}): {len(gen_files)} gen vs {len(ref_files)} ref')
        fad = FrechetAudioDistance(model_name=args.model, sample_rate=sr, use_pca=False, use_activation=False, verbose=False)
        score = float(fad.score(background_dir=str(ref_staged), eval_dir=str(gen_staged)))
    record = {'source': f'{args.gen_dir}/{args.gen_glob}', 'n_gen': len(gen_files), 'n_ref': len(ref_files), 'model': args.model, 'fad': score, 'timestamp': datetime.datetime.now().isoformat(timespec='seconds')}
    print()
    print(f'--- FAD ({args.model}) ---')
    print(f"source: {record['source']}")
    print(f"n_gen:  {record['n_gen']}")
    print(f"n_ref:  {record['n_ref']}")
    print(f"FAD:    {record['fad']:.4f}")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')
    print(f'\nappended to {RESULTS_PATH}')
if __name__ == '__main__':
    main()