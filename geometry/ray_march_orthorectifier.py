from .ray_march_dted import *
from .utils import *


class OrthoRectifier:
    def __init__(self, device, *a, **k):
        self.dtedRayMarcher = DtedRayMarcherDebugger(*a, **k, device=device)
        self.device = device

    def rectify(self, img, cam_f, cam_c, eye, R):
        d = self.device
        H,W = img.shape[:2]
        uvs = uv_grid(H,W, cam_c, cam_f, device=d)

