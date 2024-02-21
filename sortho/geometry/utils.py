import torch, numpy as np


def normalize_rows(x):
    assert x.ndim == 2
    return x / x.norm(dim=1,keepdim=True)

def grid(H,W, device=None):
    y = torch.linspace(-1,1,H, device=device)
    x = torch.linspace(-1,1,W, device=device)
    return torch.cartesian_prod(y,x).view(H,W,2).flip(-1)

def uv_grid(H,W, c, f, device=None):
    y = torch.linspace(-c[1]/f[1],c[1]/f[1],H, device=device)
    x = torch.linspace(-c[0]/f[0],c[0]/f[0],W, device=device)
    return torch.cartesian_prod(y,x).view(H,W,2).flip(-1)

def distance_along_earth_normal(p0, a, b):
    pass

def ray_march(origin, rays, depths):
    p = origin + rays * depths.view(-1,1)
    return p

