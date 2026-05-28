"""Decode cached latent tensors to waveforms via the frozen SAO VAE."""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from diffusers import AutoencoderOobleck


@torch.no_grad()
def decode_latents(latents: torch.Tensor, vae, device: torch.device,
                   batch: int = 8) -> np.ndarray:
    """latents (N, L, 64) -> waveforms (N, 2, T) float32."""
    if latents.dim() == 3 and latents.shape[-1] == 64:
        # (N, L, C) -> (N, C, L) for VAE.decode
        latents = latents.permute(0, 2, 1).contiguous()
    audio_chunks = []
    for i in range(0, latents.shape[0], batch):
        chunk = latents[i:i + batch].to(device).to(torch.bfloat16)
        out = vae.decode(chunk).sample  # (B, 2, T)
        audio_chunks.append(out.float().cpu())
    return torch.cat(audio_chunks, dim=0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", type=Path, required=True,
                    help=".pt file from sample.py (or any {ids, latents})")
    ap.add_argument("--out_dir", type=Path, required=True,
                    help="directory to write .wav files into")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading SAO VAE...")
    vae = AutoencoderOobleck.from_pretrained(
        "stabilityai/stable-audio-open-1.0", subfolder="vae"
    )
    vae = vae.to(device).to(torch.bfloat16).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    d = torch.load(args.latents, weights_only=False, map_location="cpu")
    ids = d["ids"]
    lats = d["latents"]
    print(f"decoding {lats.shape[0]} latents shape={tuple(lats.shape)}")
    audio = decode_latents(lats, vae, device, batch=args.batch)
    print(f"audio shape: {audio.shape}  range=[{audio.min():.3f},{audio.max():.3f}]")

    for i, cid in enumerate(ids):
        wav = audio[i].T  # (T, 2)
        peak = float(np.max(np.abs(wav)))
        if peak > 1e-8:
            wav = wav * (0.95 / peak)
        sf.write(args.out_dir / f"{cid}.wav", wav, args.sr, subtype="PCM_16")
    print(f"wrote {len(ids)} wavs to {args.out_dir}")


if __name__ == "__main__":
    main()
