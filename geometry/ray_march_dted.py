from frastpy2 import FlatReaderCached, EnvOptions
from utils import transform_points_epsg, Earth
import torch, numpy as np

from .utils import *

# My torch impl from ECEF -> WM is over 4x faster even on the CPU (thanks to apparent libtorch multithreading the trig calls)
# WARNING: I do get _slightly_ different results, of which I trust the libproj ones more.
USE_CUSTOM_REPROJECTION_TRANSFORM = False
USE_CUSTOM_REPROJECTION_TRANSFORM = True

'''

We need to do ray-terrain intersections (ray casting originating at the camera eye position,
                                         through the image plane at associated pixels,
                                         and onto the ground).

There's many ways of doing this.
Probably the best way I can think of is using an OpenGL renderer with LoD tile loading (as implemented by frast's "GtRenderer"), and
using the resulting depth map to simple raycast to the polygon surface using the depth values.

But that's sort of complicated.

Here is one approach that relies on Ray Marching, a very popular algorithm for the ray intersection portion of raytracers in e.g. ShaderToy shaders.
Ray Marching is consistent & safe when the Signed-Distance Function (SDF) is correct. However the implementation below's SDF is not a real SDF because
it always assumes the SDF value is the value found by looking at the terrain value directly under the current point. In reality, there may be a closer point along some diagonal line -- meaning the SDF is wrong. In practice, this hopefully does not happen much and can be ameliorated with not making a full step (using `alpha` < 1).

'''



'''
It's quite slow -- using CPU or GPU pytorch ops.
That makes me think that the GDAL transform function is slow?

TODO: Write a custom pytorch (gpu capable) EPSG 4978 -> 4326 function.

'''

class DtedRayMarcher:
    # def __init__(self, dtedPath, alpha=.9, iters=20):
    def __init__(self, dtedPath, alpha=.95, iters=40, device=None):
        eopts = EnvOptions()
        eopts.isTerrain = True
        eopts.cache = True
        self.dted = FlatReaderCached(dtedPath,eopts)
        self.alpha = alpha
        self.iters = iters
        self.device = device
        self.DT = torch.float64

    '''
    Input `eye` is in ECEF.
    `R_ltp_from_body` is ltp/enu world_from_camera.
    '''
    def march(self, eye, R_ltp_from_body, uvs):
        d = self.device
        R_ltp_from_body = R_ltp_from_body.to(self.DT)
        uvs = uvs.to(self.DT)
        eye = eye.to(self.DT)

        with torch.no_grad():
            N,two = uvs.shape
            assert two == 2


            Ltp = get_ltp(eye[None])[0]
            R = Ltp @ R_ltp_from_body # TODO: multiply by ltp to make R_ecef_from_body
            rays0 = normalize_rows(torch.cat((uvs, torch.ones_like(uvs[:,:1])), 1))
            eye, rays0, R = eye.to(d), rays0.to(d), R.to(d)
            rays = rays0 @ R.mT

            eye_wgs = torch.from_numpy(transform_points_epsg(4978, 4326, eye.cpu().numpy()[None])).to(d)

            # FIXME: Must I use this?
            scale_factor = 1./np.cos(np.deg2rad(eye_wgs[0,1].cpu()))
            print('scale_factor',scale_factor, 'eye altitude is', eye_wgs[0,2])

            # Make first sample as the elev under `eye`. Replicate.
            _, _, initial_elev = self.sample_dted(eye[None])
            initial_elev = initial_elev.flatten().mean()
            elev = torch.full((N,), initial_elev)
            # depths = eye - initial_elev
            depths = eye_wgs[0,2] - initial_elev
            # print(eye[None].shape, rays.shape, depths.shape)
            rpts = ray_march(eye[None], rays, depths * self.alpha)
            print('initial rpts moved:', (rpts-eye).norm(dim=1))

            for i in range(self.iters):
                rpts_wm, elevs = self.sample_dted_and_sample_points(rpts)
                rpts_alt = rpts_wm[:,2] #* scale_factor
                # rpts_wgs = torch.from_numpy(transform_points_epsg(4978, 4326, rpts.cpu().numpy())).to(d)
                # rpts_alt0 = rpts_wgs[:,2]
                # print(rpts_alt0-rpts_alt)
                depths = rpts_alt - elevs
                rpts1 = ray_march(rpts, rays, depths * self.alpha)
                dd = (rpts1-rpts).norm(dim=1)
                dd = dd * ((depths>0).to(self.DT) * 2 - 1) # Print negative numbers if the point is _under_ the terrain
                # print('                 relative elev', rpts_wgs[:,2], elevs)
                # print('                 depths', depths)
                print(f'                step[{i:>2d}] rpts moved:', dd)
                rpts = rpts1

                if (abs(dd) < .5).all():
                    print(f' - early stop, all changes pretty small')
                    break

        return rpts

    def sample_dted(self, ecef_positions, size=256):
        d = self.device

        from utils_torch import torch_ecef_to_wm
        if USE_CUSTOM_REPROJECTION_TRANSFORM:
            wm_positions = torch_ecef_to_wm(ecef_positions)
        else:
            wm_positions = torch.from_numpy(transform_points_epsg(4978, 3857, ecef_positions.cpu().numpy()))

        tlbr_wm = torch.cat((wm_positions[...,:2].min(0).values, wm_positions[...,:2].max(0).values))
        if (tlbr_wm[2:] - tlbr_wm[:2] < 1).any():
            tlbr_wm[:2] -= .5
            tlbr_wm[2:] += .5
        # print('sample wm',tlbr_wm, 'size', tlbr_wm[2:]-tlbr_wm[:2])
        dimg = self.dted.rasterIo(tlbr_wm.cpu().numpy(), size,size, 1)
        dimg = dimg.astype(np.float32) / 8
        dimg = torch.from_numpy(dimg)
        # print(f' - read dted range: {dimg.min()} -> {dimg.max()}')
        return wm_positions.to(d), tlbr_wm.to(d), dimg.to(d)

    def sample_dted_and_sample_points(self, ecef_positions, size=256):
        wm_positions, tlbr_wm, dted_img = self.sample_dted(ecef_positions, size)
        p = (wm_positions[:,:2] - tlbr_wm[:2][None]) / (tlbr_wm[2:] - tlbr_wm[:2])[None] * 2 - 1
        # print(dted_img.shape, p.shape)
        dted_img = dted_img.permute(2,0,1)[None] # BCHW
        p = p.view(1,-1,2)[None].float()         # NHW2, make H=1, W=N
        dted_values = torch.nn.functional.grid_sample(dted_img,p)[0,0,0] # BCHW, so just return W=N
        # print(dted_values.shape)
        return wm_positions, dted_values

