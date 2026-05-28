from __future__ import annotations
from pathlib import Path
import torch
RUNS = {'001_lmlp': 'experiments/001_drums_pilot/samples', '002_dit': 'experiments/002_drums_dit_baseline/samples', '003_no_fnnz': 'experiments/003_drums_lmlp_no_fnn_z/samples'}
STEPS = [5000, 50000, 100000]

def _load(run_dir: str, step: int) -> torch.Tensor | None:
    path = Path(run_dir) / f'latents_step_{step:07d}.pt'
    if not path.exists():
        return None
    return torch.load(path, map_location='cpu', weights_only=True).float()

def _stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = a - b
    return {'shape': tuple(a.shape), 'bit_identical': bool(torch.equal(a, b)), 'allclose_1e-6': bool(torch.allclose(a, b, atol=1e-06)), 'max_abs_diff': diff.abs().max().item(), 'mean_abs_diff': diff.abs().mean().item(), 'rel_l2': (diff.norm() / a.norm().clamp_min(1e-12)).item(), 'cosine_sim': torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()}

def main():
    labels = list(RUNS.keys())
    for step in STEPS:
        print(f'\n=== step {step} ===')
        tensors = {k: _load(v, step) for k, v in RUNS.items()}
        missing = [k for k, t in tensors.items() if t is None]
        if missing:
            print(f'  missing: {missing}; skipping step {step}')
            continue
        for i, a_lbl in enumerate(labels):
            for b_lbl in labels[i + 1:]:
                s = _stats(tensors[a_lbl], tensors[b_lbl])
                print(f"  {a_lbl} vs {b_lbl}: shape={s['shape']}")
                print(f"     bit_identical={s['bit_identical']}  allclose_1e-6={s['allclose_1e-6']}")
                print(f"     max_abs_diff={s['max_abs_diff']:.6f}  mean_abs_diff={s['mean_abs_diff']:.6f}")
                print(f"     rel_l2={s['rel_l2']:.6f}  cosine_sim={s['cosine_sim']:.6f}")
if __name__ == '__main__':
    main()