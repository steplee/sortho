import numpy as np, torch

def to_torch(*a):
    if len(a) == 1:
        b=a[0]
        return (torch.from_numpy(b) if isinstance(b,np.ndarray) else b)
    return tuple(torch.from_numpy(b) if isinstance(b,np.ndarray) else b for b in a)

def format_size(n):
    if n > (1<<30): return f'{n/(1<<30):2.1f}GB'
    if n > (1<<20): return f'{n/(1<<20):2.1f}MB'
    if n > (1<<10): return f'{n/(1<<10):2.1f}KB'
    return f'{n}B'

def get_ltp(p):
    if isinstance(p,torch.Tensor):
        assert p.ndimension() == 2, 'This function takes rank-2 inputs and returns rank-3 outputs'
        from torch.nn.functional import normalize
        up = torch.FloatTensor([0,0,1]).view(1,3).to(p.dtype).to(p.device)
        f = normalize(p)
        # r = normalize(torch.cross(f,up))
        r = normalize(torch.cross(up,f))
        u = normalize(torch.cross(f,r))
        R = torch.stack((r,u,f),2)
        return R
    else:
        assert p.ndim == 2, 'This function takes rank-2 inputs and returns rank-3 outputs'
        up = np.array([0,0,1]).reshape(1,3).astype(p.dtype)
        normalize = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
        f = normalize(p)
        r = normalize(np.cross(up,f))
        u = normalize(np.cross(f,r))
        R = np.stack((r,u,f),2)
        return R

def rodrigues(r):
    n = r.norm()
    x,y,z = r / n
    K = torch.FloatTensor((
        0, -z, y,
        z, 0, -x,
        -y, x, 0)).view(3,3)
    return torch.eye(3) + n.sin() * K + (1 - n.cos()) * (K@K)
