from __future__ import annotations
import torch
import yaml
from src.models import build_model, param_count

def main():
    cfg = yaml.safe_load(open('configs/lmlp_drums_pilot.yaml'))['model']
    cfg['seq_len'] = 43
    for arch in ['lmlp', 'dit_small']:
        cfg['arch'] = arch
        m = build_model(cfg)
        x = torch.randn(2, 43, 64)
        t = torch.rand(2)
        y = m(x, t)
        print(f'{arch}: params={param_count(m):,} out_shape={tuple(y.shape)}')
if __name__ == '__main__':
    main()