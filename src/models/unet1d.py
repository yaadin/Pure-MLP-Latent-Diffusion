"""
1D U-Net denoiser for SAO-latent audio diffusion.

Symmetric encoder/decoder with skip connections, time-conditioned via FiLM
(scale + shift produced by a small MLP from the sinusoidal timestep embedding).
Input/output: (B, L, C) where C = latent_channels (matches L-MLP / DiT).
Internally permuted to (B, D, L) for 1D conv.
"""
from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lmlp import TimestepEmbedding  # reuse


class FiLM(nn.Module):
    """Per-channel scale + shift driven by the time embedding."""

    def __init__(self, embed_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, channels * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)  t_emb: (B, embed_dim); squeeze the token dim
        if t_emb.dim() == 3:
            t_emb = t_emb.squeeze(1)
        gs = self.proj(t_emb)              # (B, 2C)
        g, s = gs.chunk(2, dim=-1)         # (B, C), (B, C)
        return x * (1.0 + g[:, :, None]) + s[:, :, None]


class ResBlock1D(nn.Module):
    """Conv -> GroupNorm -> SiLU -> FiLM -> Conv -> residual."""

    def __init__(self, in_ch: int, out_ch: int, embed_dim: int, groups: int = 8):
        super().__init__()
        g_in = min(groups, in_ch)
        g_out = min(groups, out_ch)
        self.norm1 = nn.GroupNorm(g_in, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.film  = FiLM(embed_dim, out_ch)
        self.norm2 = nn.GroupNorm(g_out, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        if in_ch != out_ch:
            self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        r = self.skip(x)
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.film(h, t_emb)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + r


class Downsample1D(nn.Module):
    """Stride-2 conv. Halves length, preserves channels."""

    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample1D(nn.Module):
    """Nearest-neighbor upsample to a target length, then conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        x = F.interpolate(x, size=target_len, mode="nearest")
        return self.op(x)


class UNet1D(nn.Module):
    """1D U-Net audio-latent denoiser.

    Args:
        latent_channels: input/output channels (e.g. 64 for SAO).
        seq_len: input sequence length L (uncondioned; used only by trainer).
        embed_dim: base width at level 0 (the time embedding lives in this dim).
        depth: total ResBlocks per level (per encoder/decoder side).
        mlp_ratio: kept for signature parity with other archs; unused here.
        channel_mults: width multipliers per level. Length defines num levels.
        blocks_per_level: number of ResBlocks at each level (encoder + decoder).
    """

    def __init__(
        self,
        latent_channels: int = 64,
        seq_len: int = 43,
        embed_dim: int = 96,
        depth: int = 6,                              # ignored; kept for build_model compatibility
        mlp_ratio: float = 2.0,                      # unused
        channel_mults: Sequence[int] = (1, 2, 2),
        blocks_per_level: int = 2,
    ):
        super().__init__()
        del depth, mlp_ratio  # not used by this architecture
        self.embed_dim = embed_dim
        self.t_embed = TimestepEmbedding(embed_dim)
        self.embed_in = nn.Conv1d(latent_channels, embed_dim, kernel_size=3, padding=1)

        widths = [embed_dim * m for m in channel_mults]
        self.widths = widths
        n_levels = len(widths)

        # Encoder: at each level, run `blocks_per_level` ResBlocks, then downsample
        # (except at the bottom level). Record skip tensors after each ResBlock.
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        cur = embed_dim
        for li, w in enumerate(widths):
            for bi in range(blocks_per_level):
                self.enc_blocks.append(ResBlock1D(cur, w, embed_dim))
                cur = w
            if li < n_levels - 1:
                self.downsamples.append(Downsample1D(cur))
            else:
                self.downsamples.append(nn.Identity())

        # Middle
        self.mid_block_a = ResBlock1D(cur, cur, embed_dim)
        self.mid_block_b = ResBlock1D(cur, cur, embed_dim)

        # Decoder: mirror of encoder. At each level, upsample, then run blocks
        # with skip concat.
        self.upsamples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        cur_dec = cur
        for li in reversed(range(n_levels)):
            w = widths[li]
            if li < n_levels - 1:
                self.upsamples.append(Upsample1D(cur_dec))
            else:
                self.upsamples.append(nn.Identity())
            for bi in range(blocks_per_level):
                # First block at this level takes (cur_dec from up) concat skip (width w)
                self.dec_blocks.append(ResBlock1D(cur_dec + w, w, embed_dim))
                cur_dec = w

        self.norm_out = nn.GroupNorm(min(8, embed_dim), embed_dim)
        self.embed_out = nn.Conv1d(embed_dim, latent_channels, kernel_size=3, padding=1)

        self.blocks_per_level = blocks_per_level
        self.n_levels = n_levels

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C). Permute to conv layout.
        x = x.transpose(1, 2)                # (B, C, L)
        t_tok = self.t_embed(t)              # (B, 1, D); TimestepEmbedding returns (B, 1, embed_dim)
        h = self.embed_in(x)                 # (B, embed_dim, L)

        # Encoder
        skips: list[torch.Tensor] = []
        lens: list[int] = []
        eb_idx = 0
        for li in range(self.n_levels):
            for bi in range(self.blocks_per_level):
                h = self.enc_blocks[eb_idx](h, t_tok)
                eb_idx += 1
                skips.append(h)
                lens.append(h.shape[-1])
            h = self.downsamples[li](h)

        # Middle
        h = self.mid_block_a(h, t_tok)
        h = self.mid_block_b(h, t_tok)

        # Decoder
        db_idx = 0
        # Decoder traverses levels top-to-bottom (reverse of encoder).
        for ui, li in enumerate(reversed(range(self.n_levels))):
            # `upsamples[ui]` brings h to the LENGTH of this level's skip tensors.
            # Take any skip from this level to find target length:
            level_start = li * self.blocks_per_level
            target_len = lens[level_start]
            if isinstance(self.upsamples[ui], Upsample1D):
                h = self.upsamples[ui](h, target_len)
            for bi in range(self.blocks_per_level):
                skip = skips[level_start + (self.blocks_per_level - 1 - bi)]
                # Right-pad/truncate skip to match h's length if off-by-1
                if skip.shape[-1] != h.shape[-1]:
                    if skip.shape[-1] > h.shape[-1]:
                        skip = skip[..., : h.shape[-1]]
                    else:
                        pad = h.shape[-1] - skip.shape[-1]
                        skip = F.pad(skip, (0, pad))
                h = torch.cat([h, skip], dim=1)
                h = self.dec_blocks[db_idx](h, t_tok)
                db_idx += 1

        h = F.silu(self.norm_out(h))
        out = self.embed_out(h)              # (B, C, L)
        return out.transpose(1, 2)           # (B, L, C)
