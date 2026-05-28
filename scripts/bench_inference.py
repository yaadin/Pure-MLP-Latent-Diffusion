"""
Inference benchmark across all 4 architectures.

Measures, for both Track A (L=43) and Track B (L=172):
  - CPU latency: single sample, single thread (deployment-relevant)
  - CPU latency: single sample, default torch threads
  - GPU latency: single sample (peak performance)
  - GPU latency: batch=8 (throughput-oriented)
  - Theoretical FLOPs per forward pass
  - Peak resident memory (process RSS during inference)

This is the "is MLP actually lightweight?" answer the paper needs.
"""
from __future__ import annotations
import argparse
import gc
import os
import time
from contextlib import contextmanager

import torch

from src.models import build_model, param_count


@contextmanager
def threads(n: int):
    """Temporarily set torch thread count."""
    prev_intra = torch.get_num_threads()
    prev_inter = torch.get_num_interop_threads()
    torch.set_num_threads(n)
    try:
        # interop can only be set once per process; best effort
        torch.set_num_interop_threads(n)
    except RuntimeError:
        pass
    try:
        yield
    finally:
        torch.set_num_threads(prev_intra)


def make_model_for_arch_and_L(arch: str, seq_len: int):
    """Build an arch + seq_len configured to match our training matrix."""
    if arch == "lmlp":
        cfg = dict(arch="lmlp", latent_channels=64, seq_len=seq_len,
                   embed_dim=128, depth=6, mlp_ratio=2.0, merge_is_mlp=True)
    elif arch == "lmlp_no_fnn_z":
        cfg = dict(arch="lmlp", latent_channels=64, seq_len=seq_len,
                   embed_dim=128, depth=6, mlp_ratio=2.0, merge_is_mlp=False)
    elif arch == "dit":
        cfg = dict(arch="dit_small", latent_channels=64, seq_len=seq_len,
                   embed_dim=128, depth=6, mlp_ratio=4.0, num_heads=4)
    elif arch == "unet":
        cfg = dict(arch="unet1d", latent_channels=64, seq_len=seq_len,
                   embed_dim=56, depth=6, mlp_ratio=2.0,
                   channel_mults=[1, 2, 2], blocks_per_level=1)
    else:
        raise ValueError(arch)
    m = build_model(cfg).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def time_inference(model, batch: int, seq_len: int, device: str,
                   n_warmup: int = 5, n_measure: int = 30) -> dict:
    """Return median, p25, p75, mean wall-clock per forward in seconds."""
    model = model.to(device)
    x = torch.randn(batch, seq_len, 64, device=device)
    t = torch.rand(batch, device=device)

    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(x, t)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    for _ in range(n_measure):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x, t)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    n = len(times)
    return {
        "mean":   sum(times) / n,
        "median": times[n // 2],
        "p25":    times[n // 4],
        "p75":    times[(3 * n) // 4],
        "min":    times[0],
    }


def count_flops_via_hooks(model, batch: int, seq_len: int) -> int:
    """Approximate FLOPs counter using forward hooks on Linear / Conv1d.
    Each Linear contributes 2 * in * out * (batch * seq_len_per_dim) FLOPs.
    Each Conv1d contributes 2 * Cin * Cout * K * out_len * batch FLOPs.
    """
    total = 0

    def linear_hook(mod, inp, out):
        nonlocal total
        x = inp[0]
        # x shape: (..., in_features); number of "spatial" positions = product of leading dims
        spatial = 1
        for s in x.shape[:-1]:
            spatial *= int(s)
        in_f = mod.in_features
        out_f = mod.out_features
        total += 2 * in_f * out_f * spatial

    def conv1d_hook(mod, inp, out):
        nonlocal total
        cin = mod.in_channels
        cout = mod.out_channels
        K = mod.kernel_size[0]
        # out shape: (B, Cout, Lout)
        if out is None:
            return
        b = int(out.shape[0])
        Lout = int(out.shape[-1])
        total += 2 * cin * cout * K * Lout * b

    handles = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, torch.nn.Conv1d):
            handles.append(m.register_forward_hook(conv1d_hook))

    x = torch.randn(batch, seq_len, 64)
    t = torch.rand(batch)
    with torch.no_grad():
        _ = model(x, t)

    for h in handles:
        h.remove()
    return total


def proc_rss_mb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/inference_bench.tsv")
    args = ap.parse_args()

    archs = ["lmlp", "lmlp_no_fnn_z", "dit", "unet"]
    seq_lens = [43, 172]

    rows = []
    print(f"{'arch':<16} {'L':>4} {'params':>10} {'flops_M':>9} "
          f"{'cpu1_ms_med':>12} {'cpu_def_ms_med':>15} "
          f"{'gpu1_ms_med':>12} {'gpu8_ms_med':>12}", flush=True)
    print("-" * 100)

    for L in seq_lens:
        for arch in archs:
            m = make_model_for_arch_and_L(arch, L)
            n_params = param_count(m)

            # FLOPs per forward at batch=1
            try:
                flops = count_flops_via_hooks(m, batch=1, seq_len=L)
            except Exception as e:
                flops = -1

            # CPU single-thread
            with threads(1):
                cpu1 = time_inference(m, 1, L, "cpu")

            # CPU default threads
            cpu_def = time_inference(m, 1, L, "cpu")

            # GPU single sample
            try:
                gpu1 = time_inference(m, 1, L, "cuda")
            except Exception:
                gpu1 = None

            # GPU batch=8
            try:
                gpu8 = time_inference(m, 8, L, "cuda")
            except Exception:
                gpu8 = None

            row = dict(
                arch=arch, L=L, params=n_params, flops=flops,
                cpu1_ms_med=cpu1["median"] * 1000,
                cpu_def_ms_med=cpu_def["median"] * 1000,
                gpu1_ms_med=(gpu1 or {"median": float('nan')})["median"] * 1000,
                gpu8_ms_med=(gpu8 or {"median": float('nan')})["median"] * 1000,
                rss_mb=proc_rss_mb(),
            )
            rows.append(row)
            print(f"{arch:<16} {L:>4} {n_params:>10,} {flops/1e6:>9.1f} "
                  f"{row['cpu1_ms_med']:>12.2f} {row['cpu_def_ms_med']:>15.2f} "
                  f"{row['gpu1_ms_med']:>12.3f} {row['gpu8_ms_med']:>12.3f}", flush=True)

            del m
            gc.collect()
            torch.cuda.empty_cache()

    # Save
    from pathlib import Path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ["arch", "L", "params", "flops",
            "cpu1_ms_med", "cpu_def_ms_med",
            "gpu1_ms_med", "gpu8_ms_med", "rss_mb"]
    with open(out, "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
