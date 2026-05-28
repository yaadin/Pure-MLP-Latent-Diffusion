from .dit_small import DiTSmall
from .lmlp import LMLPBlock, ULMLP
from .unet1d import UNet1D
__all__ = ['LMLPBlock', 'ULMLP', 'DiTSmall', 'UNet1D']

def param_count(model) -> int:
    return sum((p.numel() for p in model.parameters() if p.requires_grad))

def build_model(cfg: dict):
    arch = cfg['arch'].lower()
    common = dict(latent_channels=cfg['latent_channels'], seq_len=cfg['seq_len'], embed_dim=cfg['embed_dim'], depth=cfg['depth'], mlp_ratio=cfg['mlp_ratio'])
    if arch == 'lmlp':
        return ULMLP(**common, merge_is_mlp=cfg.get('merge_is_mlp', True))
    if arch in ('dit', 'dit_small'):
        return DiTSmall(**common, num_heads=cfg.get('num_heads', 4))
    if arch in ('unet', 'unet1d'):
        return UNet1D(
            latent_channels=cfg['latent_channels'],
            seq_len=cfg['seq_len'],
            embed_dim=cfg['embed_dim'],
            channel_mults=tuple(cfg.get('channel_mults', (1, 2, 2))),
            blocks_per_level=cfg.get('blocks_per_level', 2),
        )
    raise ValueError(f'unknown arch: {arch}')