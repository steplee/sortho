import numpy as np, cv2, os, sys
import torch

'''
The model is amazing, but not super easy to use.
I'll see about exporting a torch.jit module later...
'''
import sys, os
loftr_dir = os.path.expanduser('~/stuff/LoFTR')
assert os.path.exists(loftr_dir)
sys.path.append(loftr_dir)

from src.loftr import LoFTR, default_cfg

class LoftrMatcher:
    def __init__(self, weightsPath='/data/chromeDownloads/outdoor_ds.ckpt'):
        # default_cfg['match_coarse']['thr'] = .2
        matcher = LoFTR(config=default_cfg)
        matcher.load_state_dict(torch.load(weightsPath)['state_dict'])
        matcher = matcher.eval().cuda()
        self.matcher = matcher

    def match(self, x,y, debugShow=False):
        with torch.no_grad():
            # To support multiple batch items, have to use the 'ids' outputs

            if isinstance(x, np.ndarray): x = torch.from_numpy(x)
            if isinstance(y, np.ndarray): y = torch.from_numpy(y)
            assert x.ndimension() == 3
            assert x.dtype == torch.uint8
            x = (x.cuda().float() / 255)[None].permute(0,3,1,2)
            y = (y.cuda().float() / 255)[None].permute(0,3,1,2)
            if x.size(1) > 1: x = x.mean(1, keepdim=True)
            if y.size(1) > 1: y = y.mean(1, keepdim=True)

            d = dict(image0=x,image1=y)
            self.matcher(d)

            # For the coarse points, set output as center of cell rather than top-left
            RES = 8

            if 0:
                mkpts0 = d['mkpts0_f']
                mkpts1 = d['mkpts1_f'].cpu().numpy()
            else:
                mkpts0 = d['mkpts0_c'].long() + RES//2
                mkpts1 = d['mkpts1_c'].long() + RES//2
            mconf = d['mconf']
            sigma = RES/4 + RES * .5 / mconf

            if debugShow is not None:
                from matplotlib.cm import hsv
                h1,w1 = x.shape[-2:]
                h2,w2 = y.shape[-2:]
                hh,ww = max(h1,h2), max(w1,w2)
                pad = 32
                h,w = hh + 2*pad, ww*2 + 3*pad
                dimg = np.zeros((h,w,3),dtype=np.uint8)

                dimg[pad+hh//2-h1//2:pad+hh//2+h1//2,pad+ww//2-w1//2:pad+ww//2+w1//2] = (x[0]*255).permute(1,2,0).cpu().numpy()
                dimg[pad+hh//2-h2//2:pad+hh//2+h2//2,ww+2*pad+ww//2-w2//2:ww+2*pad+ww//2+w2//2] = (y[0]*255).permute(1,2,0).cpu().numpy()

                kpts0 = mkpts0.cpu().numpy()
                kpts1 = mkpts1.cpu().numpy()
                nconf = mconf.cpu().numpy()

                if debugShow == 'anim':
                    dimg0 = np.copy(dimg,'C')
                    limg = np.copy(dimg0,'C')*0
                    for j,(p1,p2,score) in enumerate(zip(kpts0+np.array((pad,pad))[None], kpts1+np.array((2*pad+ww,pad))[None], nconf)):
                        toint = lambda p: tuple(p.astype(int))
                        c = [int(c*255*score) for c in hsv(j/len(kpts0))[:3]]
                        cv2.line(limg, toint(p1), toint(p2), c, 1)
                        dimg = cv2.addWeighted(dimg0, .8, limg, 1, 0)
                        limg = cv2.addWeighted(limg, .94, limg, .0, 0)
                        cv2.imshow('matches', dimg)
                        cv2.waitKey(1)
                else:
                    dimg0 = np.copy(dimg,'C')
                    for j,(p1,p2,score) in enumerate(zip(kpts0+np.array((pad,pad))[None], kpts1+np.array((2*pad+ww,pad))[None], nconf)):
                        toint = lambda p: tuple(p.astype(int))
                        c = [int(c*255*score) for c in hsv(j/len(kpts0))[:3]]
                        cv2.line(dimg, toint(p1), toint(p2), c, 1)
                    cv2.imshow('matches', dimg)
                    cv2.waitKey(0)

            return dict(apts=mkpts0, bpts=mkpts1, conf=mconf, sigma=sigma)


def load(p):
    a = cv2.imread(os.path.expanduser(p), 0)
    if len(a.shape) == 2: a = a[...,None]
    h,w = a.shape[:2]
    a = torch.from_numpy(a)
    # a = a[None].permute(0,3,1,2).cuda().float()
    a = a[h//2-256:h//2+256, w//2-256:w//2+256]
    return a

if __name__ == '__main__':
    m = LoftrMatcher()
    a = torch.randn(1,1,256,256).cuda()
    b = torch.randn(1,1,256,256).cuda()
    a = load('~/Pictures/Screenshots/watergate1.png')
    b = load('~/Pictures/Screenshots/watergate2.png')
    print(m.match(a,b,debugShow=True))
