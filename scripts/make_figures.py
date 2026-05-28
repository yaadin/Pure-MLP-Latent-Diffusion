"""
Produce all paper figures from the Phase 2 matrix.

Outputs (PDF + PNG) into paper/figures/:
    loss_curves_track_a.{pdf,png}      training loss vs step, 4 archs × 3 seeds (Track A)
    loss_curves_track_b.{pdf,png}      same for Track B
    metric_bars.{pdf,png}              NN-STFT-L1 + CLAP-FAD bars, both tracks
    metric_trajectory.{pdf,png}        CLAP-FAD at 50k vs 100k per arch (convergence)
    spec_grid_track_a.{pdf,png}        spectrograms: real | each arch (Track A, 2s)
    spec_grid_track_b.{pdf,png}        spectrograms: real | each arch (Track B, 8s)
    inference_bars.{pdf,png}           CPU + GPU latency, FLOPs by arch and L
    latent_std.{pdf,png}               sample latent std vs training std (mode collapse visual)

Usage:
    python scripts/make_figures.py                # everything
    python scripts/make_figures.py --only loss    # just loss curves
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, stdev

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT  = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ----- arch presentation -----
ARCH_ORDER = ["lmlp", "lmlp_no_fnn_z", "dit", "unet"]
ARCH_LABEL = {
    "lmlp":          "L-MLP",
    "lmlp_no_fnn_z": "L-MLP (no FNN$_Z$)",
    "dit":           "DiT-small",
    "unet":          "1D U-Net",
}
ARCH_COLOR = {
    "lmlp":          "#1f77b4",   # blue
    "lmlp_no_fnn_z": "#ff7f0e",   # orange
    "dit":           "#d62728",   # red
    "unet":          "#2ca02c",   # green
}

TRACK_LABEL = {"a": "Track A (FSD50K one-shots, $L{=}43$)",
               "b": "Track B (GMD loops, $L{=}172$)"}


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.4,
    })
    return plt


def save(fig, name: str):
    for ext in ("pdf", "png"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path)
        print(f"  -> {path}")


# ---------------------------------------------------------- loss curves -----

def parse_run_id(name: str):
    m = re.match(r"(\d+)_track_([ab])_(.+?)_s(\d+)$", name)
    if not m:
        return None
    return {"exp_id": int(m.group(1)), "track": m.group(2),
            "arch": m.group(3), "seed": int(m.group(4))}


def load_train_log(exp_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, losses) arrays from train_log.jsonl."""
    log_path = exp_dir / "runs" / "train_log.jsonl"
    if not log_path.exists():
        return np.empty(0), np.empty(0)
    steps, losses = [], []
    for line in log_path.read_text().strip().split("\n"):
        try:
            d = json.loads(line)
            steps.append(d["step"]); losses.append(d["loss"])
        except Exception:
            continue
    return np.asarray(steps, dtype=np.int64), np.asarray(losses, dtype=np.float64)


def fig_loss_curves(plt, track: str):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    runs_by_arch: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for exp in sorted((REPO / "experiments").iterdir()):
        if not exp.is_dir(): continue
        meta = parse_run_id(exp.name)
        if not meta or meta["track"] != track: continue
        s, L = load_train_log(exp)
        if len(s) == 0: continue
        runs_by_arch.setdefault(meta["arch"], []).append((s, L))

    for arch in ARCH_ORDER:
        runs = runs_by_arch.get(arch, [])
        if not runs:
            continue
        # Align to a common step grid via interpolation
        ref_steps = runs[0][0]
        stacked = np.stack([np.interp(ref_steps, s, L) for (s, L) in runs])
        mu = stacked.mean(axis=0)
        lo = stacked.min(axis=0)
        hi = stacked.max(axis=0)
        ax.plot(ref_steps, mu, color=ARCH_COLOR[arch], label=ARCH_LABEL[arch])
        ax.fill_between(ref_steps, lo, hi, color=ARCH_COLOR[arch], alpha=0.15, linewidth=0)
    ax.set_xlabel("Training step")
    ax.set_ylabel("$v$-prediction MSE")
    ax.set_title(f"Training loss, {TRACK_LABEL[track]}")
    ax.set_xlim(0, 100000)
    ax.legend(loc="upper right", framealpha=0.9)
    save(fig, f"loss_curves_track_{track}")
    plt.close(fig)


# ---------------------------------------------------------- metric bars -----

def collect_metric_rows() -> list[dict]:
    rows = []
    for exp in sorted((REPO / "experiments").iterdir()):
        if not exp.is_dir(): continue
        meta = parse_run_id(exp.name)
        if not meta: continue
        for step_dir in sorted((exp / "eval").iterdir() if (exp / "eval").exists() else []):
            m = re.match(r"step_(\d+)$", step_dir.name)
            if not m: continue
            mfile = step_dir / "metrics.json"
            if not mfile.exists(): continue
            mj = json.load(open(mfile))
            ffile = step_dir / "clap_fad.json"
            clap = json.load(open(ffile))["clap_fad"] if ffile.exists() else None
            rows.append({
                **meta,
                "step": int(m.group(1)),
                "nn":   mj["nn_stft_l1"]["mean"],
                "lstd": mj["sample_latent_stats"]["std"],
                "clap": clap,
            })
    return rows


def aggregate_by(rows, track, step):
    out: dict[str, dict] = {}
    for arch in ARCH_ORDER:
        sub = [r for r in rows if r["track"] == track and r["step"] == step and r["arch"] == arch]
        if not sub:
            continue
        nn_vals  = [r["nn"]   for r in sub]
        cl_vals  = [r["clap"] for r in sub if r["clap"] is not None]
        ls_vals  = [r["lstd"] for r in sub]
        out[arch] = {
            "nn_mean":   mean(nn_vals),
            "nn_std":    (stdev(nn_vals) if len(nn_vals) > 1 else 0.0),
            "clap_mean": mean(cl_vals) if cl_vals else None,
            "clap_std":  (stdev(cl_vals) if len(cl_vals) > 1 else 0.0),
            "lstd_mean": mean(ls_vals),
        }
    return out


def fig_metric_bars(plt):
    rows = collect_metric_rows()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    width = 0.35

    for ax, metric, label in zip(axes,
                                 ["nn", "clap"],
                                 ["NN-STFT-L1 (↓)", "CLAP-FAD (↓)"]):
        archs = ARCH_ORDER
        x = np.arange(len(archs))
        # Track A
        a_data = aggregate_by(rows, "a", 100000)
        b_data = aggregate_by(rows, "b", 100000)
        a_m   = [a_data.get(a, {}).get(f"{metric}_mean", np.nan) for a in archs]
        a_e   = [a_data.get(a, {}).get(f"{metric}_std", 0)       for a in archs]
        b_m   = [b_data.get(a, {}).get(f"{metric}_mean", np.nan) for a in archs]
        b_e   = [b_data.get(a, {}).get(f"{metric}_std", 0)       for a in archs]
        ax.bar(x - width/2, a_m, width, yerr=a_e, label="Track A",
               color=[ARCH_COLOR[a] for a in archs],
               edgecolor="black", linewidth=0.4, hatch="")
        ax.bar(x + width/2, b_m, width, yerr=b_e, label="Track B",
               color=[ARCH_COLOR[a] for a in archs],
               edgecolor="black", linewidth=0.4, hatch="///", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_LABEL[a] for a in archs], rotation=20, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label.split(" ")[0] + " by architecture (3 seeds @ 100k)")
        ax.legend(loc="upper right", framealpha=0.9)

    save(fig, "metric_bars")
    plt.close(fig)


# ------------------------------------------------------ trajectory plot -----

def fig_metric_trajectory(plt):
    rows = collect_metric_rows()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

    for ax, track in zip(axes, ["a", "b"]):
        for arch in ARCH_ORDER:
            r50 = aggregate_by(rows, track, 50000).get(arch, {})
            r100 = aggregate_by(rows, track, 100000).get(arch, {})
            if "clap_mean" not in r50 or r50["clap_mean"] is None: continue
            if "clap_mean" not in r100 or r100["clap_mean"] is None: continue
            ax.plot([50_000, 100_000],
                    [r50["clap_mean"], r100["clap_mean"]],
                    marker="o", color=ARCH_COLOR[arch], label=ARCH_LABEL[arch])
            ax.fill_between([50_000, 100_000],
                            [r50["clap_mean"] - r50["clap_std"],
                             r100["clap_mean"] - r100["clap_std"]],
                            [r50["clap_mean"] + r50["clap_std"],
                             r100["clap_mean"] + r100["clap_std"]],
                            alpha=0.12, color=ARCH_COLOR[arch], linewidth=0)
        ax.set_xlabel("Training step")
        ax.set_ylabel("CLAP-FAD (↓)")
        ax.set_title(f"Convergence, {TRACK_LABEL[track]}")
        ax.set_xticks([50_000, 100_000])
        ax.set_xticklabels(["50k", "100k"])
        ax.legend(loc="upper right", framealpha=0.9)

    save(fig, "metric_trajectory")
    plt.close(fig)


# ---------------------------------------------------- spectrogram grids -----

def fig_spec_grid(plt, track: str):
    import librosa
    import librosa.display as ld
    src_root = REPO / f"results/listening_track_{track}_s0_step100k"
    if not src_root.exists():
        print(f"  spec_grid_{track}: source dir missing, skip")
        return
    rows = [("Real GMD/FSD50K", "_real_test_clips")] + [
        (ARCH_LABEL[a], a) for a in ARCH_ORDER if (src_root / a).exists()
    ]
    n_cols = 5

    fig, axes = plt.subplots(len(rows), n_cols,
                             figsize=(2.0 * n_cols, 1.6 * len(rows)),
                             squeeze=False)
    sr = 44100
    n_mels = 80
    for ri, (label, subdir) in enumerate(rows):
        wavs = sorted((src_root / subdir).glob("*.wav"))[:n_cols]
        for ci, p in enumerate(wavs):
            import soundfile as sf
            a, _ = sf.read(str(p), dtype="float32", always_2d=True)
            mono = a.mean(axis=1)
            S = librosa.feature.melspectrogram(
                y=mono, sr=sr, n_fft=2048, hop_length=512, n_mels=n_mels)
            S_db = librosa.power_to_db(S + 1e-10, ref=np.max)
            ax = axes[ri][ci]
            ax.imshow(S_db, origin="lower", aspect="auto", cmap="magma")
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(label, fontsize=8)
    for ax_row in axes:
        for ax in ax_row[len(wavs):]:
            ax.axis("off")
    fig.suptitle(f"Mel-spectrograms, {TRACK_LABEL[track]}", fontsize=10)
    save(fig, f"spec_grid_track_{track}")
    plt.close(fig)


# ---------------------------------------------------- inference benchmark --

# Hard-coded from scripts/bench_inference.py run output (params filled in by hand
# because the benchmark's auto-count returned 0; known bug, ignored here).
INFERENCE = [
    # (arch, L, params, flops_M, cpu1_ms_med, cpu_def_ms_med, gpu1_ms_med)
    ("lmlp",          43,  954_160,  73.9, 2.54, 1.95, 4.51),
    ("lmlp_no_fnn_z", 43,  855_016,  65.2, 2.31, 1.72, 3.98),
    ("dit",           43,  944_256,  70.9, 3.49, 2.24, 3.61),
    ("unet",          43,  895_168,  30.7, 2.49, 2.13, 5.64),
    ("lmlp",         172,  954_160, 324.0, 5.77, 3.35, 4.42),
    ("lmlp_no_fnn_z",172,  855_016, 290.0, 5.31, 3.14, 4.15),
    ("dit",          172,  944_256, 278.0, 9.78, 4.30, 3.81),
    ("unet",         172,  895_168, 120.4, 3.72, 2.95, 5.24),
]


def fig_inference_bars(plt):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    archs = ARCH_ORDER
    x = np.arange(len(archs))
    width = 0.35

    def col(rows, key):
        # group by arch, return [L=43, L=172]
        out = {a: {"L=43": None, "L=172": None} for a in archs}
        for r in rows:
            arch, L = r[0], r[1]
            idx = {
                "params": 2, "flops": 3, "cpu1": 4, "cpu_def": 5, "gpu1": 6,
            }[key]
            out[arch][f"L={L}"] = r[idx]
        return out

    # 1) CPU single-thread latency
    ax = axes[0]
    data = col(INFERENCE, "cpu1")
    a_vals = [data[a]["L=43"]  for a in archs]
    b_vals = [data[a]["L=172"] for a in archs]
    ax.bar(x - width/2, a_vals, width, label="$L{=}43$ (Track A)",
           color=[ARCH_COLOR[a] for a in archs], edgecolor="black", linewidth=0.4)
    ax.bar(x + width/2, b_vals, width, label="$L{=}172$ (Track B)",
           color=[ARCH_COLOR[a] for a in archs], edgecolor="black",
           linewidth=0.4, hatch="///", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABEL[a] for a in archs], rotation=20, ha="right")
    ax.set_ylabel("CPU single-thread latency (ms)")
    ax.set_title("CPU inference (1 thread, 1 sample)")
    ax.legend()

    # 2) FLOPs
    ax = axes[1]
    data = col(INFERENCE, "flops")
    a_vals = [data[a]["L=43"]  for a in archs]
    b_vals = [data[a]["L=172"] for a in archs]
    ax.bar(x - width/2, a_vals, width, label="$L{=}43$",
           color=[ARCH_COLOR[a] for a in archs], edgecolor="black", linewidth=0.4)
    ax.bar(x + width/2, b_vals, width, label="$L{=}172$",
           color=[ARCH_COLOR[a] for a in archs], edgecolor="black",
           linewidth=0.4, hatch="///", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABEL[a] for a in archs], rotation=20, ha="right")
    ax.set_ylabel("FLOPs per forward (M)")
    ax.set_title("Theoretical compute")
    ax.legend()

    # 3) GPU latency
    ax = axes[2]
    data = col(INFERENCE, "gpu1")
    a_vals = [data[a]["L=43"]  for a in archs]
    b_vals = [data[a]["L=172"] for a in archs]
    ax.bar(x - width/2, a_vals, width, label="$L{=}43$",
           color=[ARCH_COLOR[a] for a in archs], edgecolor="black", linewidth=0.4)
    ax.bar(x + width/2, b_vals, width, label="$L{=}172$",
           color=[ARCH_COLOR[a] for a in archs], edgecolor="black",
           linewidth=0.4, hatch="///", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABEL[a] for a in archs], rotation=20, ha="right")
    ax.set_ylabel("GPU latency (ms, RTX 5070, BF16)")
    ax.set_title("GPU inference (b=1)")
    ax.legend()

    save(fig, "inference_bars")
    plt.close(fig)


# ---------------------------------------------------- latent stats viz -----

def fig_latent_std(plt):
    rows = collect_metric_rows()
    # Training-set latent std for reference, per track
    train_std = {"a": 0.752, "b": 0.917}   # from build_manifest stats earlier

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    archs = ARCH_ORDER
    x = np.arange(len(archs))
    width = 0.35

    for k, track in enumerate(["a", "b"]):
        data = aggregate_by(rows, track, 100000)
        vals = [data.get(a, {}).get("lstd_mean", np.nan) for a in archs]
        ax.bar(x + (k - 0.5) * width, vals, width,
               label=f"Track {track.upper()} samples",
               color=[ARCH_COLOR[a] for a in archs],
               edgecolor="black", linewidth=0.4,
               hatch=("" if k == 0 else "///"), alpha=(1.0 if k == 0 else 0.85))
        ax.axhline(train_std[track], linestyle="--", color="black", alpha=0.4,
                   linewidth=0.8)
        ax.text(len(archs) - 0.5, train_std[track] + 0.005,
                f"Track {track.upper()} training std={train_std[track]:.2f}",
                fontsize=7, ha="right", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABEL[a] for a in archs], rotation=20, ha="right")
    ax.set_ylabel("Generated-sample latent std")
    ax.set_title("Sample variance vs training variance (mode-collapse indicator)")
    ax.legend(loc="upper right")
    save(fig, "latent_std")
    plt.close(fig)


# ---------------------------------------------------- main -----

ALL_FIGS = ["loss", "bars", "trajectory", "spec", "inference", "latent_std"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=ALL_FIGS,
                    help=f"figures to produce (default: all). Choose from {ALL_FIGS}")
    args = ap.parse_args()
    plt = setup_mpl()
    todo = set(args.only)

    if "loss" in todo:
        print("=== loss curves ===")
        fig_loss_curves(plt, "a")
        fig_loss_curves(plt, "b")

    if "bars" in todo:
        print("=== metric bars ===")
        fig_metric_bars(plt)

    if "trajectory" in todo:
        print("=== metric trajectory ===")
        fig_metric_trajectory(plt)

    if "spec" in todo:
        print("=== spectrogram grids ===")
        fig_spec_grid(plt, "a")
        fig_spec_grid(plt, "b")

    if "inference" in todo:
        print("=== inference bars ===")
        fig_inference_bars(plt)

    if "latent_std" in todo:
        print("=== latent std ===")
        fig_latent_std(plt)

    print(f"\nfigures written to {OUT}")


if __name__ == "__main__":
    main()
