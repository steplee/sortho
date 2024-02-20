import torch

def approx_resolution(gs, imgs, frameHWs):
    dx = frameHWs[:,1].view(-1,1,1) * (gs[:, :, 2:, 0] - gs[:, :, :-2, 0]) / 2
    dy = frameHWs[:,0].view(-1,1,1) * (gs[:, 2:, :, 1] - gs[:, :-2, :, 1]) / 2
    dx = torch.nn.functional.pad(dx, (1,1,0,0), mode='replicate')
    dy = torch.nn.functional.pad(dy, (0,0,1,1), mode='replicate')
    r = 1. / (dx * dy).clamp(1e-12,9e9).sqrt()
    return r


# NOTE: Zero has the special value that should be KEPT zero after any blurs or anything.
def form_masks_by_closest_to_center(srcGrids, imgs, frameHWs):
    tileSize = imgs.shape[-1]

    d = srcGrids.norm(dim=-1)

    score_d   = (-d*2).exp()
    # print(res, score_res, score_d)
    score = score_d

    if 0:
        res = approx_resolution(srcGrids, imgs, frameHWs)
        score_res = .5 / (1 + .5*abs(res-1))
        score = score + score_res


    # print('imgs',imgs.shape)
    score[(imgs==0).all(-1)] = 0
    # print('SCORE',score)

    masks = (score >= score.max(0,keepdim=True).values).float()
    masks = masks / masks.sum(0,keepdim=True)
    masks = masks.clamp(1e-6, 999)
    masks[(imgs==0).all(-1)] = 0
    masks[masks.isnan()] = 0
    # print(masks.shape, masks, score.max(0,keepdim=True).values)
    return masks

