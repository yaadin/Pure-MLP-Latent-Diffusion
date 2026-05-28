"""
Aggregate eval metrics across the Phase 2 experiment matrix into a single
TSV table + a markdown summary printable for the paper.

Reads:
    experiments/0*_track_*_*/eval/step_*/metrics.json
    experiments/1*_track_*_*/eval/step_*/metrics.json

Writes:
    results/phase2_table.tsv
    results/phase2_summary.md
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results"
OUT.mkdir(exist_ok=True)


def parse_run_id(exp_dir_name: str) -> dict | None:
    # e.g. "010_track_a_lmlp_s0" or "022_track_b_lmlp_no_fnn_z_s2"
    m = re.match(r"(\d+)_track_([ab])_(.+?)_s(\d+)$", exp_dir_name)
    if not m:
        return None
    return {
        "exp_id": int(m.group(1)),
        "track": m.group(2),
        "arch": m.group(3),
        "seed": int(m.group(4)),
    }


def parse_step(eval_subdir_name: str) -> int | None:
    m = re.match(r"step_(\d+)$", eval_subdir_name)
    return int(m.group(1)) if m else None


def collect() -> list[dict]:
    rows = []
    for exp in sorted((REPO / "experiments").iterdir()):
        if not exp.is_dir():
            continue
        meta = parse_run_id(exp.name)
        if meta is None:
            continue
        eval_root = exp / "eval"
        if not eval_root.exists():
            continue
        for step_dir in sorted(eval_root.iterdir()):
            step = parse_step(step_dir.name)
            if step is None:
                continue
            metrics_file = step_dir / "metrics.json"
            if not metrics_file.exists():
                continue
            with open(metrics_file) as f:
                m = json.load(f)
            row = dict(meta)
            row["step"] = step
            row["nn_stft_l1_mean"] = m["nn_stft_l1"]["mean"]
            row["nn_stft_l1_std"]  = m["nn_stft_l1"]["std"]
            row["latent_std"] = m["sample_latent_stats"]["std"]
            row["latent_mean"] = m["sample_latent_stats"]["mean"]
            # Optional CLAP-FAD if present in a sibling file
            fad_file = step_dir / "clap_fad.json"
            if fad_file.exists():
                with open(fad_file) as f:
                    row["clap_fad"] = json.load(f)["clap_fad"]
            else:
                row["clap_fad"] = None
            rows.append(row)
    return rows


def write_tsv(rows: list[dict], path: Path):
    keys = [
        "exp_id", "track", "arch", "seed", "step",
        "nn_stft_l1_mean", "nn_stft_l1_std", "latent_std", "latent_mean",
        "clap_fad",
    ]
    with open(path, "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")


def summary_markdown(rows: list[dict]) -> str:
    """Group by (track, arch, step), report seed-aggregate stats."""
    grp: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        grp[(r["track"], r["arch"], r["step"])].append(r)

    def fmt_mean_std(values: list[float]) -> str:
        if not values:
            return "-"
        if len(values) == 1:
            return f"{values[0]:.4f}"
        return f"{mean(values):.4f} ± {stdev(values):.4f}"

    md = ["# Phase 2 results summary\n",
          "Aggregated across seeds. Lower NN-STFT-L1 = closer to held-out test "
          "drums. CLAP-FAD shown when available.\n"]

    tracks = sorted({k[0] for k in grp})
    for track in tracks:
        md.append(f"\n## Track {track.upper()}\n")
        steps = sorted({k[2] for k in grp if k[0] == track})
        archs = sorted({k[1] for k in grp if k[0] == track})
        for step in steps:
            md.append(f"\n### Step {step:,}\n")
            md.append("| Arch | n_seeds | NN-STFT-L1 ↓ | CLAP-FAD ↓ | latent_std |")
            md.append("|---|---:|---:|---:|---:|")
            for arch in archs:
                seeds = grp.get((track, arch, step), [])
                if not seeds:
                    continue
                n = len(seeds)
                nn_vals = [r["nn_stft_l1_mean"] for r in seeds]
                fad_vals = [r["clap_fad"] for r in seeds if r["clap_fad"] is not None]
                std_vals = [r["latent_std"] for r in seeds]
                md.append(
                    f"| {arch} | {n} | "
                    f"{fmt_mean_std(nn_vals)} | "
                    f"{fmt_mean_std(fad_vals)} | "
                    f"{fmt_mean_std(std_vals)} |"
                )
    return "\n".join(md) + "\n"


def main():
    rows = collect()
    if not rows:
        print("no metrics.json files found yet; run training + eval first")
        return
    print(f"collected {len(rows)} metric rows")
    write_tsv(rows, OUT / "phase2_table.tsv")
    md = summary_markdown(rows)
    (OUT / "phase2_summary.md").write_text(md)
    print(f"wrote {OUT / 'phase2_table.tsv'}")
    print(f"wrote {OUT / 'phase2_summary.md'}")
    print("\n" + md)


if __name__ == "__main__":
    main()
