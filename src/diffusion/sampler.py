from __future__ import annotations
import torch

@torch.no_grad()
def ddim_sample(model: torch.nn.Module, shape: tuple[int, int, int], num_steps: int=50, device: torch.device | str='cpu', eta: float=0.0, seed: int | None=None) -> torch.Tensor:
    assert eta == 0.0, 'Only deterministic DDIM (eta=0) implemented for the pilot.'
    model.eval()
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(*shape, generator=gen, device=device)
    else:
        x = torch.randn(*shape, device=device)
    ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    for i in range(num_steps):
        t_cur = ts[i].expand(shape[0])
        t_nxt = ts[i + 1].expand(shape[0])
        v_pred = model(x, t_cur)
        a_cur = torch.cos(t_cur * torch.pi / 2).view(-1, 1, 1)
        s_cur = torch.sin(t_cur * torch.pi / 2).view(-1, 1, 1)
        a_nxt = torch.cos(t_nxt * torch.pi / 2).view(-1, 1, 1)
        s_nxt = torch.sin(t_nxt * torch.pi / 2).view(-1, 1, 1)
        x0_pred = a_cur * x - s_cur * v_pred
        eps_pred = s_cur * x + a_cur * v_pred
        x = a_nxt * x0_pred + s_nxt * eps_pred
    return x