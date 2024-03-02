from sortho.geometry.orthorectifier import OrthoRectifier
from sortho.utils.etc import get_ltp, to_torch
import sys, os, numpy as np, torch, cv2
from sortho.utils.q import q_to_matrix, q_mult
from sortho.utils.etc import format_size
from .masking import form_masks_by_closest_to_center
from .utils import *

def get_blender(conf):
    if conf.blend.kind == 'lap':
        from .laplacian_blend import LaplacianBlender
        return LaplacianBlender(device=torch.device('cuda:0'), **conf.blend)
    if conf.blend.kind == 'simple':
        def go(imgs, weights, **k):
            weights = weights / weights.sum(0,keepdim=True).clamp(1e-8,9999)
            return (imgs * weights).sum(0)
        return go

def unpack_frame(item):
        img = item.frame.img
        cam_f = item.frame.intrin.f
        cam_c = item.frame.intrin.c
        eye = item.posePrior.pos
        pq = item.posePrior.pq
        sq = item.frame.sq
        R_ltp_from_sensor = q_to_matrix(q_mult(pq,sq))
        return img, cam_f, cam_c, eye, R_ltp_from_sensor

class UnpackedFrameData:
    def __init__(self, fwpp):
        pass


#
#
# NOTE: Vocabulary:
#             ftd: frame-to-tilerange dict
#             tfd: tile-to-frame dict
#             framed: frame dict
#
class Blender1:
    def __init__(self, conf):
        self.conf = conf
        # self.orthor = OrthoRectifier(conf.wmPixelLevel, torch.device('cuda:0'), conf.dtedPath)
        self.orthor = OrthoRectifier(conf.wmPixelLevel, torch.device('cpu:0'), conf.dtedPath)
        self.blender = get_blender(conf)
        self.channels = conf.channels

        self.weights_blend_filter = make_gauss(21,15)


    def run(self):
        print(' - Get Frame TileRange Dict')
        ftd = self._get_frame_tilerange_dict()
        print(ftd)
        print(' - Get InvList')
        tfd = self._get_invlist(ftd)
        loader = self._get_loader(True)
        print(' - |ftd|:', len(ftd))
        print(' - |tfd|:', len(tfd))
        self._blend_tiles(tfd, loader, outputPath=self.conf.outputPath)

    def _get_loader(self, loadImages=True):
        from sortho.loading.random_access import RandomAccessLoader
        loader = RandomAccessLoader(**self.conf.input, loadImages=loadImages)
        return loader


    def _get_frame_tilerange_dict(self):
        ftd = {}
        for i,fwpp in enumerate(self._get_loader(loadImages=False).iterFramesWithPosePriors()):
            _, cam_f, cam_c, eye, R_ltp_from_sensor = unpack_frame(fwpp)
            ex = self.orthor.get_extent(cam_f, cam_c, eye, R_ltp_from_sensor)
            ftd[i] = ex
        return ftd

    # Return tile-to-frame dict from frame-to-tilerange dict
    def _get_invlist(self, ftd):
        tfd = {}
        for k, wm_tlbr in ftd.items():
            for xy in torch.cartesian_prod(
                    torch.arange(wm_tlbr[0], wm_tlbr[2]),
                    torch.arange(wm_tlbr[1], wm_tlbr[3])).numpy():
                xy = tuple(xy)
                if xy not in tfd: tfd[xy] = [k]
                else:             tfd[xy].append(k)
        return tfd

    def _rectify_one_tile_from_one_frame(self, fwpp, tile_xy, pad):
        img, cam_f, cam_c, eye, R_ltp_from_sensor = unpack_frame(fwpp)
        img, eye, cam_f, cam_c, R_ltp_from_sensor = to_torch(img, eye, cam_f, cam_c, R_ltp_from_sensor)
        Ltp = get_ltp(eye[None])[0]
        R = Ltp @ R_ltp_from_sensor
        return self.orthor.do_one_tile(img, cam_f, cam_c, eye, R, tile_xy, pad=pad)

    def _blend_weights(self, w):

        #return w # NOTE: disabled.

        # Blur but retain TRUE zeros.
        w = w.permute(0,3,1,2)
        w1 = w
        w1 = blur(self.weights_blend_filter, w1, stride=1)
        w1 = blur(self.weights_blend_filter, w1, stride=1)
        w1 = blur(self.weights_blend_filter, w1, stride=1)
        w1[w<1e-7] = 0
        w = w1.permute(0,2,3,1)
        return w

    def _blend_tiles(self, tfd, loader, tryShowAll=False, outputPath=None):
        if len(tfd) < 12*12:
            maxy = max([k[1] for k in tfd.keys()])
            maxx = max([k[0] for k in tfd.keys()])
            miny = min([k[1] for k in tfd.keys()])
            minx = min([k[0] for k in tfd.keys()])
            aimg = np.zeros(((1+maxy-miny)*256,(1+maxx-minx)*256,3),dtype=np.uint8)
        else:
            tryShowAll = False

        if outputPath:
            from .tiff_tile_writer import TiffTileWriter
            maxy = max([k[1] for k in tfd.keys()])
            maxx = max([k[0] for k in tfd.keys()])
            miny = min([k[1] for k in tfd.keys()])
            minx = min([k[0] for k in tfd.keys()])
            wmPixelTlbr = np.array((minx,miny,1+maxx,1+maxy)) * 256
            tileWriter = TiffTileWriter(outputPath, self.channels, self.conf.wmPixelLevel, wmPixelTlbr)
        else: tileWriter = None

        # TODO: multiprocess.pool this
        print('\n',' ','*'*80,sep='')
        print(' * Writing tiles *')
        print(' ','*'*80,'\n',sep='')
        # from multiprocess import Pool
        from tqdm import tqdm
        for tileCoord, frameKeys in tqdm(tfd.items()):
            frames = [loader[k] for k in frameKeys]
            pad = 16
            srcGrids, imgs = [],[]
            frameHWs = []
            for fwpp in frames:
                g,oimg = self._rectify_one_tile_from_one_frame(fwpp, torch.LongTensor(tileCoord), pad=pad)
                g,oimg = g.flip(0), oimg.flip(0)
                imgs.append(oimg)
                srcGrids.append(g)
                frameHWs.append(torch.LongTensor(fwpp.frame.img.shape[:2]))
            imgs, srcGrids = torch.stack(imgs,0), torch.cat(srcGrids,0).float()
            frameHWs = torch.stack(frameHWs,0)
            weights = form_masks_by_closest_to_center(srcGrids, imgs, frameHWs)
            weights = weights[...,None]
            weights = self._blend_weights(weights)

            from matplotlib.cm import hsv
            color = torch.from_numpy(hsv(np.linspace(0,1-1/len(weights),len(weights)))[...,:3]).to(weights.device).float().view(-1,1,1,3)
            wsum = weights.sum(0)
            wimg = (weights * color).sum(0) / wsum
            wimg[(wsum==0).all(-1)] = 0
            wimg = (wimg * 255).clamp(0,255).byte().cpu().numpy()

            yy = (tileCoord[1] - miny) * 256
            # yy = (maxy-miny - 1 - (tileCoord[1] - miny)) * 256
            xx = (tileCoord[0] - minx) * 256

            img = self.blender(imgs, weights)
            if pad > 0: img = img[pad:-pad, pad:-pad]

            if tileWriter is not None:
                if img.ndim == 2: img = img[...,None]
                if img.shape[2] == 1 and self.channels == 3: img = np.repeat(img,3,2)
                if img.shape[2] == 3 and self.channels == 1: img = img.astype(np.float32).mean(-1,keepdims=True).astype(np.uint8)
                tileWriter.writeTile(xx//256,yy//256,img)

            if tryShowAll:
                aimg[yy:yy+256, xx:xx+256] = img.cpu().numpy()#[::-1]
                cv2.imshow('weights', wimg)
                cv2.imshow('blended', aimg)
                cv2.waitKey(0)
            elif 0:
                cv2.imshow('weights', wimg)
                cv2.imshow('blended', img.cpu().numpy())
                cv2.waitKey(0)



def run(conf):
    blender = Blender1(conf)
    blender.run()

if __name__ == '__main__':
    from omegaconf import OmegaConf
    confs = []

    if 1:
        print('WARNING: fixed tmp config')
        confs = [OmegaConf.create(dict(
            input=dict(path='/data/inertialLabs/flightFeb15/sortho.ra', frameStride=1, firstFrame=0),
            dtedPath='/data/elevation/srtm/srtm.fft',
            wmPixelLevel=24,
            blend=dict(kind='lap', k=9, sigma=3.5, nlvl=5),
            outputPath='/tmp/tst.tif',
            channels=1,
            # blend=dict(kind='simple'),
            # blend=dict(kind='lap', k=15, sigma=4.5, nlvl=5),
            ))]
        confs.append(OmegaConf.from_cli())

    else:
        if len(sys.argv) > 1 and '=' not in sys.argv[1]:
            confs.append(OmegaConf.load(sys.argv[1]))
            sys.argv.pop(1)
        confs.append(OmegaConf.from_cli())

    conf = OmegaConf.merge(*confs)
    run(conf)
