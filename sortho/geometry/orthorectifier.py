from .ray_march_dted import *
from sortho.utils.geo import *
from sortho.utils.torch_geo import torch_ecef_to_wm, torch_wm_to_ecef

PIXEL_TO_TILE_OFFSET = 8
TILE_SIZE = (1<<PIXEL_TO_TILE_OFFSET)

'''

Ortho-rectify camera frames.

Summary:

    This works by determining the WM tiles needed to capture all projected pixels from a frame.
    A grid of pixels is formed for all output pixels. The grid is initially formed of 2d WM coordinates. The grid is
    given Z values by sampling DTED at those WM coordinates. Then this 3d grid is transformed to ECEF.

    The ECEF points are projected onto the image plane. Now we have a map from output (orthorectified) pixels to pixels of the frame.
    We bilinearly sample the input frame with these projected positions to form the output image.

Estimating frame extent:

    The first step of orthorectification will be to find the minimum set of WM tiles on the configured level
    that are fully cover the projected pixels of the frame.

    For this, I use the `DtedRayMarcher` class.
    This is a bit overkill, but I implemented it thinking it could be as the main engine for orthorectification, only to see that it is
    more useful for the inverse operation! (i.e. usefule for rendering an already orthorectified image, but not producing it)

    Still using it here is legitimate and better than alternatives. The clearest alternative would be doing 1 shot ray-plane intersection
    based on the dted value sampled under the camera. Ray marching is like doing this but making several iterations, making it much more accurate
    in hilly scenarios.

'''

# NOTE: Mind the flip()s


def get_projection_matrix(cam_f, cam_c, eye, R, device):
    K = torch.FloatTensor((
        cam_f[0], 0, cam_c[0],
        0, cam_f[1], cam_c[1],
        0, 0, 1)).reshape(3,3).to(device)
    V = torch.cat((R.T, (-R.T@eye).view(3,1)), 1) # 3x4
    return K @ V # 3x4

def get_projection_matrix_parts(cam_f, cam_c, eye, R, device):
    K = torch.DoubleTensor((
        cam_f[0], 0, cam_c[0],
        0, cam_f[1], cam_c[1],
        0, 0, 1)).reshape(3,3).to(device)
    return K, R.to(device), eye.to(device)

class OrthoRectifier:
    def __init__(self, wmPixelLevel, device, *a, **k):
        k.setdefault('tol', 5) # Set relatively high tolerance. Ray marching needn't be very accurate.
        self.dtedRayMarcher = DtedRayMarcher(*a, **k, device=torch.device('cpu'))
        self.dted = self.dtedRayMarcher.dted
        self.device = device
        self.wmPixelLevel = wmPixelLevel

    def get_extent(self, cam_f, cam_c, eye, R_ltp_from_sensor):
        d = self.dtedRayMarcher.device

        eye, cam_f, cam_c, R_ltp_from_sensor = to_torch(eye, cam_f, cam_c, R_ltp_from_sensor)

        # could go as low as 2x2, but it's possible an interior point of the image lies outside convex hull of corners.
        partial_uvs = uv_grid(4,4, cam_c, cam_f, device=d).reshape(-1,2)
        rpts_ecef = self.dtedRayMarcher.march(eye, R_ltp_from_sensor, partial_uvs)

        rpts_wm = torch_ecef_to_wm(rpts_ecef)

        wm_tiles = Earth.decimal2integral(rpts_wm, self.wmPixelLevel-PIXEL_TO_TILE_OFFSET, cast=False)[...,:2]
        wm_tiles_tl = wm_tiles.min(0).values.floor().long()
        wm_tiles_br = wm_tiles.max(0).values.ceil().long()
        return torch.cat((wm_tiles_tl, wm_tiles_br))

    def orthorectify_tiles(self, img, cam_f, cam_c, eye, R_ltp_from_sensor):
        d = self.device

        R_ltp_from_sensor = torch.from_numpy(R_ltp_from_sensor)
        eye = torch.from_numpy(eye)
        cam_f = torch.from_numpy(cam_f)
        cam_c = torch.from_numpy(cam_c)
        img = torch.from_numpy(img)
        H,W = img.shape[:2]

        Ltp = get_ltp(eye[None])[0]
        R = Ltp @ R_ltp_from_sensor

        # wm_tiles = Earth.decimal2integral(rpts_wm, self.wmPixelLevel-PIXEL_TO_TILE_OFFSET, cast=False)[...,:2]
        wm_tiles_tlbr = self.get_extent(cam_f, cam_c, eye, R_ltp_from_sensor).cpu()
        wm_tiles_tl, wm_tiles_br = wm_tiles_tlbr[:2], wm_tiles_tlbr[2:]

        wm_tiles_sz = (wm_tiles_br - wm_tiles_tl)
        wm_tiles_n = wm_tiles_sz[0]*wm_tiles_sz[1]

        print(wm_tiles_tl, '->', wm_tiles_br)
        print(f' - Need {wm_tiles_n} tiles ({wm_tiles_sz[0]}, {wm_tiles_sz[1]})')

        assert wm_tiles_n < 200, 'too many tiles'

        #
        # `do_one_tile` supports padding (which will be useful for blending while individual tiles)
        # Unless trying to process individual tiles, `do_several_tiles` is faster and should be used
        #
        for i in range(2):
            if i==0:
                dimg = torch.zeros((TILE_SIZE*wm_tiles_sz[1], TILE_SIZE*wm_tiles_sz[0], 3), dtype=torch.uint8)

                for tile_xy in torch.cartesian_prod(
                        torch.arange(wm_tiles_tl[0], wm_tiles_br[0]),
                        torch.arange(wm_tiles_tl[1], wm_tiles_br[1])).to(d):
                    pad = 8 # NOTE: Testing pad
                    # pad = 0 # NOTE: Testing pad
                    _,timg = self.do_one_tile(img, cam_f, cam_c, eye, R, tile_xy, pad=pad)
                    ly = wm_tiles_sz[1] - 1 - (tile_xy[1] - wm_tiles_tl[1])
                    # ly = tile_xy[1] - wm_tiles_tl[1]
                    lx = tile_xy[0] - wm_tiles_tl[0]
                    h,w = TILE_SIZE,TILE_SIZE
                    if pad > 0:
                        timg = timg[pad:-pad,pad:-pad]
                    dimg[ly*h:(ly+1)*h, lx*w:(lx+1)*w] = timg.byte()
            else:
                dimg = self.do_several_tiles(img, cam_f, cam_c, eye, R, wm_tiles_tl, wm_tiles_br).byte()


            if dimg.shape[-1] != 3 or dimg.ndimension() == 2:
                dimg = dimg.view(dimg.size(0),dimg.size(1),1).repeat(1,1,3)
            green = torch.FloatTensor([0,80,0]).view(1,3).to(dimg.device)
            for y in range(0, dimg.shape[0], TILE_SIZE):
                dimg[y-2] = dimg[y-2] // 3 * 2 + green
                dimg[y+2] = dimg[y+2] // 3 * 2 + green
            for x in range(0, dimg.shape[1], TILE_SIZE):
                dimg[:,x-2] = dimg[:,x-2] // 3 * 2 + green
                dimg[:,x+2] = dimg[:,x+2] // 3 * 2 + green

            import cv2
            cv2.imshow('tiled', dimg.clamp(0,255).cpu().numpy())
            cv2.waitKey(100)
            # exit()

    '''
    Return tuple:
        `g`:    source pixel coordinates (normalized [-1,1])
        `oimg`: warped/rectified tile

    NOTE: `oimg` is a float32 tensor with zeros where no pixels matched and valid black pixels mapped to `1e-5`
          This can be used for further masking-of-valid pixels without a seperate mask tensor.

    WARNING: I probably should NOT flip the Y axes here, but I do...

    '''
    def do_one_tile(self, img, cam_f, cam_c, eye, R, tile_xy, pad=0):
        d = self.device
        pix_xy = tile_xy * TILE_SIZE
        pix_tl = Earth.integral2decimal(pix_xy-pad, self.wmPixelLevel)
        pix_br = Earth.integral2decimal(pix_xy+TILE_SIZE+pad, self.wmPixelLevel)
        size = TILE_SIZE + 2*pad

        grid_11 = grid(size,size,device=d)
        if 1: grid_11 = grid_11 * (size-2)/(size-1) # NOTE: This makes do_one_tile EXACTLY the same as do_several_tiles, which is wanted!
        grid_wm = ((grid_11 + 1) * .5 * (pix_br - pix_tl) + pix_tl).view(-1,2)

        grid_wm_elev = torch.from_numpy(self.dted.rasterIo(torch.cat((pix_tl,pix_br)).cpu().numpy(), size, size, 1).astype(np.float32)).to(d)
        grid_wm_elev = grid_wm_elev.flip(0)
        grid_wm_elev = grid_wm_elev.view(-1,1) / 8
        grid_wm = torch.cat((grid_wm, grid_wm_elev), -1)
        if USE_CUSTOM_REPROJECTION_TRANSFORM:
            grid_ecef = torch_wm_to_ecef(grid_wm)
        else:
            grid_ecef = torch.from_numpy(transform_points_epsg(3857, 4978, grid_wm.cpu().numpy())).to(d)

        # P = get_projection_matrix(cam_f,cam_c,eye,R,self.device)
        K,R,e = get_projection_matrix_parts(cam_f,cam_c,eye,R,self.device)
        proj_pts = (grid_ecef - e.view(-1,3)) @ R @ K.mT
        proj_pts = (proj_pts[...,:2] / proj_pts[...,2:]).to(torch.float32)

        img_sz = torch.FloatTensor((img.shape[1], img.shape[0])).to(d)
        g = (proj_pts / img_sz) * 2 - 1
        g = g.reshape(size, size, 2)[None]

        img = img.permute(2,0,1).to(d).float()[None].clamp(1e-5,999) # BCHW
        oimg = torch.nn.functional.grid_sample(img, g)[0].permute(1,2,0) # HWC

        # Hard set any pixels outside of the acceptable border mask to 0.0 rather than 1e-5.
        # This marks that they are invalid (we don't want to bilinear sample invalid/black and valid pixels)
        gg = (g*.5+.5) * img_sz
        border_pad = 3
        if border_pad > 0:
            border_mask = (gg[0,...,0]>img_sz[0]-border_pad) | \
                        (gg[0,...,0]<border_pad) | \
                        (gg[0,...,1]>img_sz[1]-border_pad) | \
                        (gg[0,...,1]<border_pad)
            # print('BORDER_MASK % keep',(border_mask).float().mean())
            # oimg[border_mask] //= 2
            oimg[border_mask] = 0

        oimg = oimg.flip(0)
        g = g.flip(0)
        return g,oimg

    def do_several_tiles(self, img, cam_f, cam_c, eye, R, wm_tiles_tl, wm_tiles_br):
        nx,ny = (wm_tiles_br - wm_tiles_tl).long().cpu().numpy()
        dimg = torch.zeros((TILE_SIZE*ny, TILE_SIZE*nx, 3), dtype=torch.uint8)

        d = self.device
        tile_tl = Earth.integral2decimal(wm_tiles_tl, self.wmPixelLevel-PIXEL_TO_TILE_OFFSET).to(d)
        tile_br = Earth.integral2decimal(wm_tiles_br, self.wmPixelLevel-PIXEL_TO_TILE_OFFSET).to(d)

        size = torch.LongTensor((TILE_SIZE*nx,TILE_SIZE*ny)).to(device=d)
        grid_11 = grid(size[1],size[0],device=d)
        if 1: grid_11 = grid_11 * (size-2)/(size-1)
        grid_wm = ((grid_11 + 1) * .5 * (tile_br - tile_tl) + tile_tl).view(-1,2)
        # grid_wm = ((grid(TILE_SIZE*ny,TILE_SIZE*nx,device=d) + 1) * .5 * (tile_br - tile_tl) + tile_tl).view(-1,2)

        grid_wm_elev = torch.from_numpy(self.dted.rasterIo(torch.cat((tile_tl,tile_br)).cpu().numpy(), TILE_SIZE*nx, TILE_SIZE*ny, 1).astype(np.float32)).to(d)
        grid_wm_elev = grid_wm_elev.flip(0)
        grid_wm_elev = grid_wm_elev.view(-1,1) / 8
        grid_wm = torch.cat((grid_wm, grid_wm_elev), -1)
        if USE_CUSTOM_REPROJECTION_TRANSFORM:
            grid_ecef = torch_wm_to_ecef(grid_wm)
        else:
            grid_ecef = torch.from_numpy(transform_points_epsg(3857, 4978, grid_wm.cpu().numpy())).to(d)

        # P = get_projection_matrix(cam_f,cam_c,eye,R,self.device)
        K,R,e = get_projection_matrix_parts(cam_f,cam_c,eye,R,self.device)
        proj_pts = (grid_ecef - e.view(-1,3)) @ R @ K.mT
        proj_pts = (proj_pts[...,:2] / proj_pts[...,2:]).to(torch.float32)

        img_sz = torch.FloatTensor((img.shape[1], img.shape[0])).to(d)
        g = (proj_pts / img_sz) * 2 - 1
        g = g.reshape(TILE_SIZE*ny, TILE_SIZE*nx, 2)[None]

        img = img.permute(2,0,1).to(d).float()[None].clamp(1e-5,999) # BCHW
        oimg = torch.nn.functional.grid_sample(img, g)[0].permute(1,2,0) # HWC
        oimg = oimg.flip(0)
        return oimg




if __name__ == '__main__':
    wmPixelLevel = 24
    orth = OrthoRectifier(wmPixelLevel, torch.device('cuda:0'), '/data/elevation/srtm/srtm.fft')

    from sortho.loading.terrapixel import TerraPixelLoader
    from sortho.utils.q import q_to_matrix, q_mult

    tpl = TerraPixelLoader('/data/inertialLabs/flightFeb15/irnOutput/1707947224/eval.bin', loadImages=True)
    for item in tpl:
        print(item.posePrior, item.frame.intrin)

        img = item.frame.img
        cam_f = item.frame.intrin.f
        cam_c = item.frame.intrin.c
        eye = item.posePrior.pos
        pq = item.posePrior.pq
        sq = item.frame.sq
        R_ltp_from_sensor = q_to_matrix(q_mult(pq,sq))
        # R_ltp_from_sensor = np.diag([1,-1,-1.])
        import cv2
        print(eye)
        # cv2.imshow('img',img)
        orth.orthorectify_tiles(img, cam_f, cam_c, eye, R_ltp_from_sensor)
