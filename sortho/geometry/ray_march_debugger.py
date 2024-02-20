from sortho.utils.geo import *
from .ray_march_dted import *

'''
This shows a color dset sampled from the world positions of ray-casted UV rays.
(This is sort of like the inverse operation of ortho-rectification)
'''

class DtedRayMarcherDebugger(DtedRayMarcher):
    def __init__(self, colorDsetPath, *a, **k):
        super().__init__(*a, **k)

        eopts = EnvOptions()
        eopts.isTerrain = False
        eopts.cache = True
        eopts.readonly = True
        self.colorDset = FlatReaderCached(colorDsetPath,eopts)

    def marchAndShow(self, H=512, W=512, *a, **k):
        rpts_ecef = super().march(*a, **k)
        self.show1(rpts_ecef, H, W)

    def show1(self, rpts_ecef, H, W):
        rpts_wgs  = torch.from_numpy(transform_points_epsg(4978, 4326, rpts_ecef.cpu().numpy()))
        rpts_wm   = torch.from_numpy(transform_points_epsg(4978, 3857, rpts_ecef.cpu().numpy())[...,:2]).to(self.device)

        import cv2

        cimg = self.sample_color_and_sample_points(rpts_wm, H,W)
        print(cimg.shape)
        cimg = cv2.cvtColor(cimg.cpu().numpy(), cv2.COLOR_RGB2BGR)

        elev = rpts_wgs[:,2].reshape(H,W).cpu().numpy()
        eimg = (elev - elev.min()) / (elev.max() - elev.min()) * 255
        eimg = np.copy(eimg.astype(np.uint8),'C')
        eimg = cv2.cvtColor(eimg, cv2.COLOR_GRAY2BGR)

        # return
        img = np.hstack((eimg,cimg))
        cv2.imshow('elev+color', img)
        cv2.waitKey(0)

        for i in range(1000):
            t = np.sin(i/100 * 3.141 * 5) * .5 + .5
            a = cv2.addWeighted(eimg, t, cimg, 1-t, 0)
            cv2.imshow('elev+color', a)
            cv2.waitKey(100)


    def sample_color(self, wm_positions, H, W, oversampleRatio=None):
        d = self.device
        tlbr_wm = torch.cat((wm_positions.min(0).values, wm_positions.max(0).values))
        if (tlbr_wm[2:] - tlbr_wm[:2] < 1).any():
            tlbr_wm[:2] -= .5
            tlbr_wm[2:] += .5

        # TODO: Make S bigger if skewed a lot
        if oversampleRatio is None: oversampleRatio = 1.5
        S = int(max(H,W) * oversampleRatio + .5)
        ex = tlbr_wm[2:] - tlbr_wm[:2]
        aspect_wh = ex[0] / ex[1]
        if ex[0] > ex[1]:
            WW = S
            HH = int(.5 + aspect_wh*S)
        else:
            HH = S
            WW = int(.5 + (1/aspect_wh)*S)

        cimg = self.colorDset.rasterIo(tlbr_wm.cpu().numpy(), WW,HH, 3)
        cimg = torch.from_numpy(cimg)
        # cimg = cimg.flip(0)
        return tlbr_wm.to(d), cimg.to(d)

    def sample_color_and_sample_points(self, wm_positions, H, W):
        tlbr_wm, cimg = self.sample_color(wm_positions, H,W)
        tlbr_wm = tlbr_wm.to(self.DT).to(wm_positions.device)
        cimg = cimg.to(wm_positions.device)

        p = (wm_positions - tlbr_wm[:2][None]) / (tlbr_wm[2:] - tlbr_wm[:2])[None] * 2 - 1
        cimg = cimg.permute(2,0,1)[None].float()                             # BCHW
        p = p.view(H,W,2)[None].float()                                      # NHW2
        cimg = torch.nn.functional.grid_sample(cimg,p)[0].permute(1,2,0) # BCHW -> HWC
        return cimg.byte()





def test_ltp():
    eye = torch.FloatTensor((1058033.54733714, -4829809.60757495,  4016914.60834785))

    # Check ltp vs my drawing.
    Ltp = get_ltp(torch.FloatTensor([-1,0,0]).view(1,3))[0]
    print(Ltp,Ltp.det())
    print(Ltp @ Ltp.mT)

    Ltp = get_ltp(eye[None])[0]
    print(Ltp,Ltp.det())
    print(Ltp @ Ltp.mT)
    exit()

def test_basic():
    eye = torch.FloatTensor((1058033.54733714, -4829809.60757495,  4016914.60834785))
    R = torch.eye(3) * torch.FloatTensor([1,-1,-1]).reshape(-1,3)
    uvs = torch.FloatTensor((0,0, .1, 0)).reshape(-1,2)

    dtm = DtedRayMarcher('/data/elevation/srtm/srtm.fft')
    dtm.march(eye, R, uvs)

def test_img():
    # Grand Canyon
    eye = torch.from_numpy(transform_points_epsg(4326, 4978, np.array((-112.122949, 36.092791, 25_000))[None]))[0].float()
    # SF
    eye = torch.from_numpy(transform_points_epsg(4326, 4978, np.array((-122.451933, 37.753946, 45_000))[None]))[0].float()

    R = torch.eye(3) * torch.FloatTensor([1,-1,-1]).reshape(-1,3)
    # R = rodrigues(torch.FloatTensor((np.pi,.9,0)))
    print(R)

    uvs = torch.FloatTensor((0,0, .1, 0)).reshape(-1,2)
    f = .5/np.tan(np.deg2rad(40)/2)
    H,W = 512,512
    uvs = grid(H,W).view(-1,2) * f

    if 0:
        # dtm = DtedRayMarcher('/data/elevation/srtm/srtm.fft', device=torch.device('cuda:0'))
        dtm = DtedRayMarcher('/data/elevation/srtm/srtm.fft')
        rpts = dtm.march(eye, R, uvs)

        if 0:
            depth = (rpts - eye[None]).norm(dim=1).view(H,W)
            dimg = (depth - depth.min()) / (depth.max() - depth.min()) * 255
            dimg = np.copy(dimg.byte().cpu().numpy(),'C')
            print('depth min max',depth.min(), depth.max())
            import cv2
            cv2.imshow('depth', dimg)
            cv2.waitKey(0)

        if 1:
            elev = transform_points_epsg(4978, 4326, rpts.cpu().numpy().astype(np.float64))[:,2].reshape(H,W)
            dimg = (elev - elev.min()) / (elev.max() - elev.min()) * 255
            dimg = np.copy(dimg.astype(np.uint8),'C')
            print('elev min max',elev.min(), elev.max())
            import cv2
            cv2.imshow('elev', dimg)
            cv2.waitKey(0)


def test_img2():
    # Catoctin
    eye = torch.from_numpy(transform_points_epsg(4326, 4978, np.array((-77.435169, 39.583242, 15_000))[None]))[0].float()

    # R = torch.eye(3) * torch.FloatTensor([1,-1,-1]).reshape(-1,3)
    R = rodrigues(torch.FloatTensor((np.pi*1.1,0,0)))

    uvs = torch.FloatTensor((0,0, .1, 0)).reshape(-1,2)
    f = .5/np.tan(np.deg2rad(50)/2)
    print(f)
    H,W = 512,512
    uvs = grid(H,W).view(-1,2) / f

    if 1:
        # dtm = DtedRayMarcherDebugger('/data/naip/mdpa/mdpa.fft', '/data/elevation/srtm/srtm.fft', device=torch.device('cuda:0'))
        dtm = DtedRayMarcherDebugger('/data/naip/mdpa/mdpa.fft', '/data/elevation/srtm/srtm.fft')
        rpts = dtm.marchAndShow(H,W,eye, R, uvs)

if __name__ == '__main__':

    if 0:
        test_ltp()

    if 0:
        test_basic()

    if 0:
        test_img()

    if 1:
        test_img2()


