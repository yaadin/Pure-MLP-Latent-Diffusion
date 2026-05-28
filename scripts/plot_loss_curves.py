from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
RUNS = [('L-MLP (FNN_Z = MLP)', 'experiments/001_drums_pilot/runs/train_log.jsonl', 'C0'), ('DiT-small (transformer)', 'experiments/002_drums_dit_baseline/runs/train_log.jsonl', 'C1'), ('L-MLP (FNN_Z = linear)', 'experiments/003_drums_lmlp_no_fnn_z/runs/train_log.jsonl', 'C2')]
OUT_PATH = Path('paper/figures/loss_curves.png')

def _load_log(path: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            steps.append(r['step'])
            losses.append(r['loss'])
    return (steps, losses)

def _smooth(xs: list[float], window: int=20) -> list[float]:
    out = []
    s = 0.0
    q: list[float] = []
    for x in xs:
        q.append(x)
        s += x
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=150)
    for label, path, color in RUNS:
        if not Path(path).exists():
            print(f'skip {label}: {path} not found')
            continue
        steps, losses = _load_log(path)
        smoothed = _smooth(losses, window=20)
        ax.plot(steps, losses, color=color, alpha=0.18, linewidth=0.8)
        ax.plot(steps, smoothed, color=color, linewidth=1.6, label=label)
    ax.set_xlabel('training step')
    ax.set_ylabel('v-prediction loss (MSE)')
    ax.set_title('Training loss, three matched-parameter denoisers on the drum-latent task')
    ax.set_xlim(0, 100000)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_PATH)
    print(f'wrote {OUT_PATH}')
if __name__ == '__main__':
    main()