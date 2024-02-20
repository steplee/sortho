import torch, torch.nn.functional as F
import numpy as np

def make_gauss(k, sigma):
    assert k % 2 == 1
    x = torch.arange(-k//2,k//2).float()
    g = torch.cartesian_prod(x,x)
    g = g/sigma
    g = (g*g).sum(-1)
    g = (-g).exp()
    g = g / g.sum()
    print(f' - Gaussian Kernel (size {k}x{k}), (max/min {g.max()/g.min()}) (min {g.min()}) (sum {g.sum()})')
    return g.view(1,1,k,k)

def make_upsample_kernel(k):
    assert k == 4 or k == 2
    if k == 2:
        K = torch.FloatTensor((
            1,1,
            1,1,
            )).view(1,1,2,2)
    elif k == 4:
        K = torch.FloatTensor((
            1,2,2,1,
            2,4,4,2,
            2,4,4,2,
            1,1,1,1,
            )).view(1,1,4,4)
    return 4 * (K / K.sum())
    # return 1 * (K / K.sum())

if 0:
    a = torch.zeros((1,1,4,4))
    a[0,0,1,1] = 1
    a = F.interpolate(a, scale_factor=2, mode='bilinear')
    print(a, a.sum())
    exit()

def my_upsample(K, img):
    if K is None:
        return F.interpolate(img, scale_factor=2, mode='bilinear')

    k = K.size(-1)
    assert k % 2 == 0
    return F.conv_transpose2d(img, K, padding=(k-1)//2, groups=K.size(0), stride=2)

def blur(K, img, stride=1):
    k = K.size(-1)
    return F.conv2d(img, K, padding=k//2, groups=K.size(0), stride=stride)

def gauss_pyramid(K, img, nlvl=4):
    B,C,H,W = img.shape
    out = [img]
    for i in range(nlvl-1):
        # img = blur(K, img,)
        # img = F.avg_pool2d(img, 2, 2)
        # img = F.avg_pool2d(img, 3, 2, padding=1)
        # img = img[..., ::2, ::2]
        img = blur(K, img, stride=2)
        out.append(img)
    return out


def lap_pyramid(K, img, nlvl=4, UK=None):
    # See Frame 66 of https://web.stanford.edu/class/cs231m/lectures/lecture-5-stitching-blending.pdf
    B,C,H,W = img.shape
    out = []
    for i in range(nlvl):
        if i == nlvl-1:
            out.append(img)
        else:
            img0 = img
            img = blur(K, img)
            # img = F.avg_pool2d(img, 2, 2)
            # img = F.avg_pool2d(img, 3, 2, padding=1)
            # img = img[..., ::2, ::2]
            img = blur(K, img, stride=2)
            out.append(img0 - my_upsample(UK, img))
    return out
