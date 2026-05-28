from __future__ import annotations
import torch
import torch.nn as nn

class SAOVAE(nn.Module):
    LATENT_CHANNELS = 64
    LATENT_RATE_HZ = 21.5
    SAMPLE_RATE = 44100
    AUDIO_CHANNELS = 2
    HF_REPO = 'stabilityai/stable-audio-open-1.0'
    HF_SUBFOLDER = 'vae'

    def __init__(self, dtype: torch.dtype=torch.float32, device: str | torch.device='cpu'):
        super().__init__()
        from diffusers import AutoencoderOobleck
        vae = AutoencoderOobleck.from_pretrained(self.HF_REPO, subfolder=self.HF_SUBFOLDER, torch_dtype=dtype)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        self.vae = vae.to(device)

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, sample_mode: str='mode') -> torch.Tensor:
        assert waveform.dim() == 3 and waveform.shape[1] == self.AUDIO_CHANNELS, f'expected (B, 2, T) stereo waveform, got {tuple(waveform.shape)}'
        out = self.vae.encode(waveform)
        dist = out.latent_dist
        z = dist.mode() if sample_mode == 'mode' else dist.sample()
        return z.transpose(1, 2).contiguous()

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        assert latent.dim() == 3 and latent.shape[-1] == self.LATENT_CHANNELS, f'expected (B, L, 64) channel-last latent, got {tuple(latent.shape)}'
        z = latent.transpose(1, 2).contiguous()
        return self.vae.decode(z).sample