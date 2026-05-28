from __future__ import annotations
import torch
import torch.nn as nn
from einops import rearrange

class LMLPBlock(nn.Module):

    def __init__(self, seq_len: int, embed_dim: int, mlp_ratio: float=4.0, merge_is_mlp: bool=True):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.norm_d = nn.LayerNorm(embed_dim)
        self.norm_l = nn.LayerNorm(seq_len)
        self.fnn_l = nn.Linear(seq_len, seq_len, bias=False)
        self.fnn_r = nn.Linear(embed_dim, embed_dim, bias=False)
        if merge_is_mlp:
            self.fnn_z = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        else:
            self.fnn_z = nn.Linear(embed_dim, embed_dim)
        self.norm_c = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.fnn_c = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm = rearrange(x, 'b l d -> b d l')
        x_normed = self.norm_d(x)
        x_perm_normed = self.norm_l(x_perm)
        left = self.fnn_l(x_perm_normed)
        left = rearrange(left, 'b d l -> b l d')
        right = self.fnn_r(x_normed)
        x = x + self.fnn_z(left + right)
        x = x + self.fnn_c(self.norm_c(x))
        return x

class TimestepEmbedding(nn.Module):

    def __init__(self, embed_dim: int, hidden: int | None=None):
        super().__init__()
        hidden = hidden or embed_dim * 4
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(nn.Linear(embed_dim, hidden), nn.SiLU(), nn.Linear(hidden, embed_dim))

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.embed_dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0, device=t.device)) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < self.embed_dim:
            emb = torch.nn.functional.pad(emb, (0, self.embed_dim - emb.shape[-1]))
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t))[:, None, :]

class ULMLP(nn.Module):

    def __init__(self, latent_channels: int=64, seq_len: int=45, embed_dim: int=128, depth: int=6, mlp_ratio: float=2.0, merge_is_mlp: bool=True):
        super().__init__()
        assert depth % 2 == 0, 'depth must be even (encoder/decoder symmetric)'
        self.depth = depth
        self.embed_dim = embed_dim
        self.embed_in = nn.Linear(latent_channels, embed_dim)
        self.embed_out = nn.Linear(embed_dim, latent_channels)
        self.t_embed = TimestepEmbedding(embed_dim)
        internal_seq_len = seq_len + 1
        self.encoder = nn.ModuleList([LMLPBlock(internal_seq_len, embed_dim, mlp_ratio, merge_is_mlp) for _ in range(depth // 2)])
        self.decoder = nn.ModuleList([LMLPBlock(internal_seq_len, embed_dim, mlp_ratio, merge_is_mlp) for _ in range(depth // 2)])
        self.skip_proj = nn.ModuleList([nn.Linear(embed_dim * 2, embed_dim) for _ in range(depth // 2)])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.embed_in(x)
        t_tok = self.t_embed(t)
        h = torch.cat([t_tok, h], dim=1)
        skips = []
        for block in self.encoder:
            h = block(h)
            skips.append(h)
        for block, proj in zip(self.decoder, self.skip_proj):
            skip = skips.pop()
            h = proj(torch.cat([h, skip], dim=-1))
            h = block(h)
        h = h[:, 1:, :]
        return self.embed_out(h)