import numpy as np, cv2, os, sys
import torch

import sys, os
loftr_dir = os.path.expanduser('~/stuff/LoFTR')
assert os.path.exists(loftr_dir)
sys.path.append(loftr_dir)

from src.loftr import LoFTR, default_cfg

class LoftrMatcher:
    def __init__(self, weightsPath='/data/chromeDownloads/outdoor_ds.ckpt'):
        matcher = LoFTR(config=default_cfg)
        matcher.load_state_dict(torch.load(weightsPath)['state_dict'])
        matcher = matcher.eval().cuda()
        self.matcher = matcher

    def match(self, x,y):
        # To support multiple batch items, have to use the 'ids' outputs
        assert x.size(0) == 1

        d = dict(image0=x,image1=y)
        self.matcher(d)

        RES = 8


        # mkpts0 = d['mkpts0_f'].cpu().numpy()
        # mkpts1 = d['mkpts1_f'].cpu().numpy()
        mkpts0 = d['mkpts0_c'].long().cpu().numpy() + RES//2
        mkpts1 = d['mkpts1_c'].long().cpu().numpy() + RES//2
        mconf = d['mconf'].cpu().numpy()
        sigma = 8 * .5 / mconf
        return mkpts0, mkpts1, mconf, sigma


if __name__ == '__main__':
    m = LoftrMatcher()
    a = torch.randn(1,1,256,256).cuda()
    b = torch.randn(1,1,256,256).cuda()
    print(m.match(a,b)[0])
