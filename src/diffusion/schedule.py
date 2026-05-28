from __future__ import annotations
import torch

def v_target(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = torch.cos(t * torch.pi / 2).view(-1, 1, 1)
    sigma = torch.sin(t * torch.pi / 2).view(-1, 1, 1)
    x_t = alpha * x0 + sigma * noise
    v_t = alpha * noise - sigma * x0
    return (x_t, v_t)

def sample_timesteps(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.rand(batch_size, device=device)