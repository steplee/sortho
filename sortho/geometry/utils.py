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

def get_ltp(p):
    assert p.ndimension() == 2, 'This function takes rank-2 inputs and returns rank-3 outputs'
    from torch.nn.functional import normalize
    up = torch.FloatTensor([0,0,1]).view(1,3).to(p.dtype).to(p.device)
    f = normalize(p)
    # r = normalize(torch.cross(f,up))
    r = normalize(torch.cross(up,f))
    u = normalize(torch.cross(f,r))
    R = torch.stack((r,u,f),2)
    return R

def rodrigues(r):
    n = r.norm()
    x,y,z = r / n
    K = torch.FloatTensor((
        0, -z, y,
        z, 0, -x,
        -y, x, 0)).view(3,3)
    return torch.eye(3) + n.sin() * K + (1 - n.cos()) * (K@K)
