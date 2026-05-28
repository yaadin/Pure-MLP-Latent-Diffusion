"""
Generate Phase 2 training configs for all (arch, track, seed) combinations.

Output: configs/track_{a,b}_{lmlp,lmlp_no_fnn_z,dit,unet}_s{0,1,2}.yaml

Architectures are matched at ~900k-950k params at base sequence length:
    L-MLP            954,160
    L-MLP no-FNN_Z   855,016
    DiT-small        944,256
    UNet1D            895,168
"""
from __future__ import annotations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs"
OUT.mkdir(exist_ok=True)

TRACKS = {
    "a": dict(
        latents_file="data/latents/track_a_latents.pt",
        split_file="data/manifests/track_a_splits.json",
        split_dataset="fsd50k",
        seq_len=43,
        clip_seconds=2.0,
        batch_size=64,
    ),
    "b": dict(
        latents_file="data/latents/track_b_latents.pt",
        split_file="data/manifests/track_b_splits.json",
        split_dataset="gmd",
        seq_len=172,
        clip_seconds=8.0,
        batch_size=32,
    ),
}

ARCH_BASE = {
    "lmlp": dict(
        arch="lmlp", embed_dim=128, depth=6, mlp_ratio=2.0, merge_is_mlp=True,
    ),
    "lmlp_no_fnn_z": dict(
        arch="lmlp", embed_dim=128, depth=6, mlp_ratio=2.0, merge_is_mlp=False,
    ),
    "dit": dict(
        arch="dit_small", embed_dim=128, depth=6, mlp_ratio=4.0, num_heads=4,
    ),
    "unet": dict(
        arch="unet1d", embed_dim=56, depth=6, mlp_ratio=2.0,
        channel_mults=[1, 2, 2], blocks_per_level=1,
    ),
}

# Experiment IDs are stable: 0XX = Track A, 1XX = Track B
ID_BASE = {
    "a": {"lmlp": 10, "lmlp_no_fnn_z": 12, "dit": 11, "unet": 13},
    "b": {"lmlp": 20, "lmlp_no_fnn_z": 22, "dit": 21, "unet": 23},
}

SEEDS = [0, 1, 2]


def yaml_dump(d: dict, indent: int = 0) -> str:
    """Minimal YAML emitter; handles nested dicts and lists of scalars."""
    out = []
    for k, v in d.items():
        prefix = "  " * indent
        if isinstance(v, dict):
            out.append(f"{prefix}{k}:")
            out.append(yaml_dump(v, indent + 1))
        elif isinstance(v, list):
            inner = ", ".join(repr(x) if isinstance(x, str) else str(x) for x in v)
            out.append(f"{prefix}{k}: [{inner}]")
        elif isinstance(v, bool):
            out.append(f"{prefix}{k}: {'true' if v else 'false'}")
        elif isinstance(v, str):
            # Don't quote unless needed
            needs_quote = any(c in v for c in ":,#'\"") or v.startswith(" ")
            out.append(f"{prefix}{k}: {repr(v) if needs_quote else v}")
        elif v is None:
            out.append(f"{prefix}{k}: null")
        else:
            out.append(f"{prefix}{k}: {v}")
    return "\n".join(out)


def make_config(track: str, arch: str, seed: int) -> dict:
    tcfg = TRACKS[track]
    acfg = dict(ARCH_BASE[arch])
    eid = ID_BASE[track][arch]
    exp_name = f"{eid:03d}_track_{track}_{arch}_s{seed}"
    cfg = dict(
        experiment_id=exp_name,
        seed=seed,
        data=dict(
            latents_file=tcfg["latents_file"],
            split_file=tcfg["split_file"],
            split_dataset=tcfg["split_dataset"],
            split_key="train",
            seq_len=tcfg["seq_len"],
            clip_seconds=tcfg["clip_seconds"],
        ),
        model=dict(latent_channels=64, **acfg),
        train=dict(
            device="cuda",
            bf16_autocast=True,
            batch_size=tcfg["batch_size"],
            num_workers=2,
            num_steps=100000,
            optimizer="adamw",
            lr=2.0e-4,
            weight_decay=0.03,
            betas=[0.9, 0.9],
            warmup_steps=1000,
            log_every=200,
            ckpt_every=10000,
            sample_every=10000,
        ),
        sampling=dict(num_steps=50, num_samples=8),
        paths=dict(
            run_dir=f"experiments/{exp_name}/runs",
            ckpt_dir=f"experiments/{exp_name}/checkpoints",
            samples_dir=f"experiments/{exp_name}/samples",
        ),
    )
    return cfg


def main():
    n = 0
    for track in TRACKS:
        for arch in ARCH_BASE:
            for seed in SEEDS:
                cfg = make_config(track, arch, seed)
                path = OUT / f"track_{track}_{arch}_s{seed}.yaml"
                path.write_text(yaml_dump(cfg) + "\n")
                n += 1
    print(f"wrote {n} configs to {OUT}")


if __name__ == "__main__":
    main()
