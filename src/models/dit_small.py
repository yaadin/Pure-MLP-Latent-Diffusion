from __future__ import annotations
import torch
import torch.nn as nn
from .lmlp import TimestepEmbedding

class TransformerBlock(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, bias=False)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, bias=False)
        self.norm2 = nn.LayerNorm(embed_dim, bias=False)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class DiTSmall(nn.Module):

    def __init__(self, latent_channels: int=64, seq_len: int=43, embed_dim: int=128, depth: int=6, num_heads: int=4, mlp_ratio: float=4.0):
        super().__init__()
        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'
        self.embed_dim = embed_dim
        self.embed_in = nn.Linear(latent_channels, embed_dim)
        self.embed_out = nn.Linear(embed_dim, latent_channels)
        self.t_embed = TimestepEmbedding(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm_final = nn.LayerNorm(embed_dim, bias=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.embed_in(x)
        t_tok = self.t_embed(t)
        h = torch.cat([t_tok, h], dim=1)
        h = h + self.pos_embed[:, :h.shape[1], :]
        for block in self.blocks:
            h = block(h)
        h = self.norm_final(h)
        h = h[:, 1:, :]
        return self.embed_out(h)