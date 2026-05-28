"""DDIM sampling from a trained checkpoint.

Loads a config + checkpoint, runs the deterministic DDIM sampler at the
config's sampling settings, optionally producing many more samples than the
training-time `cfg.sampling.num_samples`.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import yaml

from src.diffusion import ddim_sample
from src.models import build_model


def load_model_from_ckpt(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = ckpt["cfg"]
    model_cfg = dict(cfg["model"])
    model_cfg["seq_len"] = cfg["data"]["seq_len"]
    model = build_model(model_cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model, cfg, int(ckpt.get("step", 0))


@torch.no_grad()
def sample_latents(
    model: torch.nn.Module,
    cfg: dict,
    num_samples: int,
    device: torch.device,
    seed: int = 0,
    batch: int = 16,
) -> torch.Tensor:
    L = cfg["data"]["seq_len"]
    C = cfg["model"]["latent_channels"]
    steps = cfg["sampling"]["num_steps"]
    outs: list[torch.Tensor] = []
    done = 0
    bi = 0
    while done < num_samples:
        b = min(batch, num_samples - done)
        z = ddim_sample(
            model,
            shape=(b, L, C),
            num_steps=steps,
            device=device,
            seed=seed + bi,
        )
        outs.append(z.cpu())
        done += b
        bi += 1
    return torch.cat(outs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out",  type=Path, required=True,
                    help=".pt file to write {ids, latents} into")
    ap.add_argument("--n",    type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, step = load_model_from_ckpt(args.ckpt, device)
    print(f"loaded ckpt @ step {step:,}  arch={cfg['model']['arch']}  device={device}")

    z = sample_latents(model, cfg, args.n, device, seed=args.seed, batch=args.batch)
    ids = [f"sample_{i:05d}" for i in range(args.n)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": ids, "latents": z, "ckpt_step": step,
                "cfg": cfg, "seed": args.seed}, args.out)
    print(f"wrote {args.n} latents shape={tuple(z.shape)} -> {args.out}")


if __name__ == "__main__":
    main()
