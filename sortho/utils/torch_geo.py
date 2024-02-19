import torch, numpy as np
from .geo import Earth

'''

Even on the CPU these pytorch functions are way faster than GDAL (libproj?) coordinate transforms.
I take it that libtorch does parallel processing with the arithmetic and trignometric calls because
I get 953% cpu usage with these transforms but only 223% with the GDAL ones...

That's pretty cool.
You'd expect the overhead of python and of non-optimizable python control flow for all of the little arithmetic operations
to result in slower code. But I'm guessing the trig calls dominate and they are parallelized, so the whole thing is faster here.
(Unless the gdal/libproj calls are slower for some other, bad code related reason)

It's about 4x faster in runtime as well.

# WARNING: I do get _slightly_ different results, of which I trust the libproj ones more.

'''

# Copied from my old C++ impl. Relies on Bowrings formula IIRC. Other way to do it is with newton/root-finding
def torch_ecef_to_geodetic_rad(p):
    assert p.size(1) == 3
    p = p / Earth.R1

    xx,yy,zz = p.T

    x = torch.atan2(yy,xx)

    k = 1 / (1 - Earth.e2)
    k3 = k*k*k
    z = zz
    z2 = z*z
    p2 = xx*xx + yy*yy
    p = p2.sqrt()
    for j in range(2):
        c_i = (((1-Earth.e2)*z2) * (k*k) + p2).pow(1.5) / Earth.e2
        k = (c_i + (1-Earth.e2) * z2 * k3) / (c_i - p2)
    y = torch.atan2(k*z,p)

    rn = Earth.a / (1 - Earth.e2 * y.sin().pow(2)).sqrt()
    sinabslat = abs(y).sin()
    coslat = y.cos()
    z = (abs(z) + p - rn * (coslat + (1-Earth.e2) * sinabslat)) / (coslat + sinabslat)

    return torch.stack((x,y,z), 1)

def torch_geodetic_to_uwm(llh):
    x = llh[:,0] / np.pi
    y = (np.pi/4 + llh[:,1]*.5).tan().log() / np.pi
    z = llh[:,2] #/ np.pi
    return torch.stack((x,y,z),1)

def torch_uwm_to_geodetic_rad(p):
    x = p[:,0] * np.pi
    y = (p[:,1] * np.pi).exp().atan() * 2 - np.pi/2
    z = p[:,2] #* np.pi
    return torch.stack((x,y,z),1)

def torch_geodetic_to_ecef(llh):
    '''
    const double aa = llh[i * 3 + 0], bb = llh[i * 3 + 1], cc = llh[i * 3 + 2];
    double       cos_phi = std::cos(bb), cos_lamb = std::cos(aa);
    double       sin_phi = std::sin(bb), sin_lamb = std::sin(aa);
    double       n_phi = a / std::sqrt(1 - e2 * sin_phi * sin_phi);

    out[i * 3 + 0] = (n_phi + cc) * cos_phi * cos_lamb;
    out[i * 3 + 1] = (n_phi + cc) * cos_phi * sin_lamb;
    out[i * 3 + 2] = (b2_over_a2 * n_phi + cc) * sin_phi;
    '''

    cos_lamb, sin_lamb = llh[:,0].cos(), llh[:,0].sin()
    cos_phi,  sin_phi  = llh[:,1].cos(), llh[:,1].sin()
    n_phi = Earth.a / (1 - Earth.e2 * sin_phi * sin_phi).sqrt()

    x = (n_phi + llh[:,2]) * cos_phi * cos_lamb
    y = (n_phi + llh[:,2]) * cos_phi * sin_lamb
    z = (Earth.b2_over_a2 * n_phi + llh[:,2]) * sin_phi

    out = torch.stack((x,y,z),1)
    out = out * Earth.R1
    return out

def torch_ecef_to_uwm(p):
    return torch_geodetic_to_uwm(torch_ecef_to_geodetic_rad(p))
def torch_ecef_to_wm(p):
    return torch_geodetic_to_uwm(torch_ecef_to_geodetic_rad(p)) * torch.FloatTensor((Earth.WebMercatorScale, Earth.WebMercatorScale, Earth.R1)).to(p.device).view(1,3)

def torch_wm_to_ecef(p):
    p = p / torch.FloatTensor((Earth.WebMercatorScale, Earth.WebMercatorScale, Earth.R1)).to(p.device).view(1,3)
    return torch_geodetic_to_ecef(torch_uwm_to_geodetic_rad(p))

# Simple test to make sure it works. Also eval float64 vs float32.
if __name__ == '__main__':
    from sortho.utils.geo import Earth, transform_points_epsg

    pts0 = np.array((-77.435169, 39.583242, 15_000))[None]
    print('pts0', pts0)

    for DT in (torch.float32, torch.float64):
        print(' - with dtype', DT)
        pts1 = torch.from_numpy(transform_points_epsg(4326, 4978, pts0)).to(DT)
        pts2 = torch_ecef_to_geodetic_rad(pts1) * torch.FloatTensor((180/np.pi, 180/np.pi, Earth.R1))[None]

        print('pts1', pts1.cpu().numpy())
        print('pts2', pts2.cpu().numpy())
        print('err', (pts2-torch.from_numpy(pts0).to(DT)).norm(dim=1).mean())

        pts_wm_true = torch.from_numpy(transform_points_epsg(4326, 3857, pts0)).double()
        pts_wm_pred = torch_ecef_to_wm(pts1)
        print('pts_wm_true', pts_wm_true)
        print('pts_wm_pred', pts_wm_pred)
        print('err', (pts_wm_true - pts_wm_pred).norm(dim=1).mean())
        print('')


    # Test wm/ecef
    print(' - Test wm -> ecef\n')
    for DT in (torch.float32, torch.float64):
        print(' - with dtype', DT)
        pts1 = torch.from_numpy(transform_points_epsg(4326, 3857, pts0)).to(DT)
        pts_ecef_true = torch.from_numpy(transform_points_epsg(3857, 4978, pts1.cpu().numpy())).to(DT)
        pts_pred = torch_wm_to_ecef(pts1)
        print('err', (pts_pred-pts_ecef_true).norm(dim=1).mean())
