from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.data import LatentDataset, default_collate
from src.diffusion import ddim_sample, sample_timesteps, v_target
from src.models import build_model, param_count

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


class SyntheticLatents(Dataset):
    """Fake latents for `--smoke`. Returns shape (L, C) per item."""

    def __init__(self, num: int, seq_len: int, channels: int, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self.data = torch.randn(num, seq_len, channels, generator=gen)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def lr_lambda(step: int, warmup: int) -> float:
    if warmup <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup)


def resolve_device(requested: str) -> torch.device:
    """Map config-requested device to one that's actually present, with fallback."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("cuda requested but not available; falling back to cpu.", flush=True)
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        print("mps requested but not available; falling back to cpu.", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def build_dataset(cfg: dict) -> Dataset:
    data_cfg = cfg["data"]
    return LatentDataset(
        latents_file=data_cfg["latents_file"],
        split_file=data_cfg.get("split_file"),
        split_key=data_cfg.get("split_key", "train"),
        split_dataset=data_cfg.get("split_dataset"),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--smoke", action="store_true",
                   help="200 steps on synthetic data; no checkpointing or sampling.")
    p.add_argument("--num_synthetic", type=int, default=512,
                   help="Number of fake latents for --smoke (irrelevant otherwise).")
    p.add_argument("--max_steps_override", type=int, default=None,
                   help="Override cfg train.num_steps (useful for quick verification).")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 0))
    device = resolve_device(cfg["train"]["device"])
    use_bf16 = bool(cfg["train"].get("bf16_autocast", False)) and device.type == "cuda"

    # ----- model -----
    model_cfg = dict(cfg["model"])
    model_cfg["seq_len"] = cfg["data"]["seq_len"]
    model = build_model(model_cfg).to(device)
    n_params = param_count(model)
    print(f"model={model_cfg['arch']}  params={n_params:,}  device={device}  bf16_autocast={use_bf16}")

    # ----- data -----
    if args.smoke:
        ds: Dataset = SyntheticLatents(
            num=args.num_synthetic,
            seq_len=cfg["data"]["seq_len"],
            channels=cfg["model"]["latent_channels"],
            seed=cfg.get("seed", 0),
        )
        steps = 200
        log_every = 25
        ckpt_every = 0
        sample_every = 0
        print(f"smoke mode: synthetic dataset n={len(ds)}, steps={steps}")
    else:
        ds = build_dataset(cfg)
        steps = args.max_steps_override or cfg["train"]["num_steps"]
        log_every = cfg["train"]["log_every"]
        ckpt_every = cfg["train"]["ckpt_every"]
        sample_every = cfg["train"].get("sample_every", 0)
        print(f"dataset: {len(ds)} cached latents from {cfg['data']['latents_file']}")

    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 0),
        collate_fn=default_collate,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    if len(loader) == 0:
        raise SystemExit("Dataloader is empty; check batch_size and dataset size.")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        betas=tuple(cfg["train"]["betas"]),
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: lr_lambda(s, cfg["train"]["warmup_steps"])
    )

    # ----- io paths -----
    if not args.smoke:
        run_dir = Path(cfg["paths"]["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = Path(cfg["paths"]["ckpt_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        samples_dir = Path(cfg["paths"]["samples_dir"])
        samples_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train_log.jsonl"
        log_f = open(log_path, "a", buffering=1)
    else:
        log_f = None

    # ----- train loop -----
    model.train()
    step = 0
    t0 = time.time()
    data_iter = iter(loader)
    losses_window: list[float] = []
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32

    while step < steps:
        try:
            x0 = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x0 = next(data_iter)
        x0 = x0.to(device, non_blocking=True)
        t = sample_timesteps(x0.shape[0], device=device)
        noise = torch.randn_like(x0)
        x_t, v_t = v_target(x0, noise, t)

        if use_bf16:
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                v_pred = model(x_t, t)
                loss = torch.nn.functional.mse_loss(v_pred, v_t)
        else:
            v_pred = model(x_t, t)
            loss = torch.nn.functional.mse_loss(v_pred, v_t)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        losses_window.append(loss.item())
        step += 1

        if step % log_every == 0 or step == 1:
            avg = sum(losses_window) / len(losses_window)
            losses_window.clear()
            elapsed = time.time() - t0
            rec = {
                "step": step,
                "loss": avg,
                "lr": opt.param_groups[0]["lr"],
                "elapsed_s": round(elapsed, 1),
                "steps_per_s": round(step / max(elapsed, 1e-6), 2),
            }
            print(json.dumps(rec))
            if log_f:
                log_f.write(json.dumps(rec) + "\n")

        if ckpt_every and step % ckpt_every == 0:
            torch.save(
                {"model": model.state_dict(), "step": step, "cfg": cfg},
                ckpt_dir / f"step_{step:07d}.pt",
            )

        if sample_every and step % sample_every == 0:
            model.eval()
            with torch.no_grad():
                z = ddim_sample(
                    model,
                    shape=(
                        cfg["sampling"]["num_samples"],
                        cfg["data"]["seq_len"],
                        cfg["model"]["latent_channels"],
                    ),
                    num_steps=cfg["sampling"]["num_steps"],
                    device=device,
                    seed=cfg.get("seed", 0) + step,
                )
            torch.save(z.cpu(), samples_dir / f"latents_step_{step:07d}.pt")
            model.train()

    if log_f:
        log_f.close()
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
