import torch, torch.nn.functional as F
import numpy as np
from .utils import *



class LaplacianBlender:
    def __init__(self, k, sigma, device, nlvl=4, kind='lap'):
        assert kind == 'lap'
        self.device = device
        self.K = make_gauss(k, sigma).to(device)
        self.nlvl = nlvl

        # NOTE: Control F.interpolate vs conv2d_transpose
        self.UK = None
        # self.UK = make_upsample_kernel(2).to(device)
        # self.UK = make_upsample_kernel(4).to(device)

    def __call__(self, imgs, weights, debugShow=False):
        Kc = self.K
        UKc = self.UK
        if imgs.size(-1) != 1: Kc = Kc.repeat(imgs.size(-1), 1, 1, 1)
        if imgs.size(-1) != 1 and UKc is not None: UKc = UKc.repeat(imgs.size(-1), 1, 1, 1)

        with torch.no_grad():
            x = imgs.permute(0,3,1,2).float().to(self.device)
            w = weights.permute(0,3,1,2).float().to(self.device)
            pyr_x = lap_pyramid(Kc,x, nlvl=self.nlvl, UK=UKc)
            pyr_w = gauss_pyramid(self.K,w, nlvl=self.nlvl)
            pyr_w = [p / p.sum(0,keepdim=True).clamp(1e-8,9e5) for p in pyr_w]

            if 1:
                pyr_y = [(x*w).sum(0,keepdim=True) for x,w in zip(pyr_x, pyr_w)]
                z = pyr_y[-1]
                for yy in pyr_y[:-1][::-1]:
                    z = my_upsample(UKc,z) + yy
            else:
                size = x.shape[-2], x.shape[-1]
                y = [F.interpolate((x*w).sum(0,keepdim=True), size, mode='bilinear') for x,w in zip(pyr_x, pyr_w)]
                z = sum(y)

            if debugShow:
                n = pyr_x[0].shape[0]
                ww = sum(p.shape[-1] for p in pyr_x)
                hh = pyr_x[0].shape[-2]
                ximg = np.zeros((n*hh,ww,pyr_x[0].shape[-3]), dtype=np.uint8)
                wimg = np.zeros((n*hh,ww,pyr_w[0].shape[-3]), dtype=np.uint8)
                bimg = np.zeros((  hh,ww,pyr_y[0].shape[-3]), dtype=np.uint8)
                ox = 0
                for p in pyr_x:
                    p = (p * .5) + (255 * .5)
                    for j in range(len(p)):
                        ximg[j*hh:j*hh+p.shape[-2],ox:(ox+p.shape[-1])] = p[j].permute(1,2,0).byte().cpu().numpy()
                    ox += p.shape[-1]
                ox = 0
                for p in pyr_w:
                    for j in range(len(p)):
                        wimg[j*hh:j*hh+p.shape[-2],ox:(ox+p.shape[-1])] = (p[j]*255).permute(1,2,0).byte().cpu().numpy()
                    ox += p.shape[-1]
                ox = 0
                for p in pyr_y:
                    p=(p[0] * .5) + (255 * .5)
                    bimg[:p.shape[-2],ox:(ox+p.shape[-1])] = (p).permute(1,2,0).byte().cpu().numpy()
                    ox += p.shape[-1]
                cv2.imshow('pyr_x', ximg)
                cv2.imshow('pyr_w', wimg)
                cv2.imshow('pyr_blend', bimg)
                cv2.waitKey(0)


            return z[0].permute(1,2,0).clamp(0,255).byte()




if __name__ == '__main__':
    import cv2,os

    # k, sigma = 9, 3.5
    # k, sigma = 7, 2.5
    # k, sigma = 3, 1.
    k, sigma = 9, 3.
    # k, sigma = 17, 19.
    nlvl = 4
    b = LaplacianBlender(k, sigma, torch.device('cuda:0'), nlvl=nlvl)

    img1 = torch.zeros((512,512,3),dtype=torch.uint8).to(b.device)
    img1[256-15:256+15, 256-15:256+15] = 220
    img1[:,10] = 100
    img1[10] = 100

    img2 = torch.zeros((512,512,3),dtype=torch.uint8).to(b.device)
    img2[256-30:256+30, 256-30:256+30] = 255
    img2[:,10:13] = 100
    img2[10:13] = 100

    img1 = cv2.imread(os.path.expanduser('~/Pictures/Screenshots/apple.png'))
    img2 = cv2.imread(os.path.expanduser('~/Pictures/Screenshots/orange.png'))
    img1 = cv2.resize(img1, (256,256))
    img2 = cv2.resize(img2, (256,256))
    img1 = torch.from_numpy(img1)
    img2 = torch.from_numpy(img2)

    S = img1.shape[0]

    if 0:
        imgb = blur(b.K, img1.permute(2,0,1)[None].float())[0].permute(1,2,0).clamp(0,255).byte()
        print(img1.shape)
        print(imgb.shape)

        dimg = np.hstack((img1.cpu().numpy(),imgb.cpu().numpy()))
        cv2.imshow('blurred', dimg)
        cv2.waitKey(0)

    if 0:
        x = img1.float().permute(2,0,1)[None]
        pyr = lap_pyramid(b.K, x)
        ww = sum(p.shape[-1] for p in pyr)
        hh = pyr[0].shape[-2]
        dimg = np.zeros((hh,ww,pyr[0].shape[-3]), dtype=np.uint8)
        ox = 0
        for p in pyr:
            p = (p * .5) + (255 * .5)
            dimg[:p.shape[-2],ox:(ox+p.shape[-1])] = p[0].permute(1,2,0).byte().cpu().numpy()
            ox += p.shape[-1]

        cv2.imshow('blurred', dimg)
        cv2.waitKey(0)

    if 1:
        imgs = torch.stack((img1,img2), 0)

        # indicator = (torch.cartesian_prod(torch.arange(S),torch.arange(S)).sum(-1) > S)
        indicator = (torch.cartesian_prod(torch.arange(S),torch.arange(S))[...,1] > S/2)
        weights = torch.stack((indicator==False,indicator==True), 0)[:,None].resize(2,S,S,1).to(b.device).float()
        weights = weights.permute(0,3,1,2)
        weights = blur(make_gauss(25,15).cuda(), weights)
        if 0:
            weights = blur(make_gauss(25,9).cuda(), weights)
            weights = blur(make_gauss(25,9).cuda(), weights)
            weights = blur(make_gauss(25,9).cuda(), weights)
            weights = blur(make_gauss(25,9).cuda(), weights)
        weights = weights.float().to(b.K.device).permute(0,2,3,1)
        # weights = 1-weights

        imgb = b(imgs,weights, debugShow=True).cpu().numpy()

        dimg = np.hstack((img1.cpu().numpy(), img2.cpu().numpy(), imgb))
        cv2.imshow('blended', dimg)
        cv2.waitKey(0)
