import numpy as np, cv2, torch
import os, sys
try:
    import gdal, osr
except:
    from osgeo import gdal, gdalconst, osr


def get_transform_for_wm_tlbr(lvl, iwmTlbr):
    from sortho.utils.geo import Earth
    tl, br = Earth.integral2decimal(iwmTlbr.reshape(2,2), lvl).astype(np.float64)
    # print(tl,br)
    # scale = (br - tl) / (iwmTlbr[2:] - iwmTlbr[:2])
    # scale = (iwmTlbr[2:] - iwmTlbr[:2]) / (br - tl)
    # scale = (1/Earth.wmLevels[lvl],)*2
    scale = (Earth.wmLevels[lvl],)*2
    off = tl
    '''
    A = np.array((
        scale[0], 0, off[0],
        0, scale[1], off[1]), dtype=np.float64)
    return A
    '''
    return [
            off[0], scale[0], 0,
            off[1], 0, scale[1],
            ] # Wtf is this layout

class TiffTileWriter:
    TILE_SIZE = 256

    # def __init__(self, path, c, w, h, wmPixelLevel, wmTlbr):
    def __init__(self, path, c, wmPixelLevel, wmPixelTlbr):
        wmTlbr = np.array(wmPixelTlbr)
        h = wmTlbr[3] - wmTlbr[1]
        w = wmTlbr[2] - wmTlbr[0]
        assert w % TiffTileWriter.TILE_SIZE == 0
        assert h % TiffTileWriter.TILE_SIZE == 0

        self.path = path
        self.dset = gdal.Open(path, gdal.GA_Update)
        opts = {
            'COMPRESS': 'JPEG',
            'BLOCKXSIZE': '256',
            'BLOCKYSIZE': '256',
            'TILED': 'YES',
            'BIGTIFF': 'YES',
            'JPEG_QUALITy': '80',
        }
        driver = gdal.GetDriverByName("GTiff")
        self.dset = driver.Create(path, int(w), int(h), c, gdal.GDT_Byte, options=[k+':'+v for k,v in opts.items()])

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        gt = get_transform_for_wm_tlbr(wmPixelLevel, wmTlbr)
        self.dset.SetProjection(srs.ExportToWkt())
        self.dset.SetGeoTransform(gt)

    def writeTile(self, tx, ty, img):
        if isinstance(img, torch.Tensor): img = img.cpu().numpy()
        self.dset.WriteArray(img.transpose(2,0,1), int(tx)*TiffTileWriter.TILE_SIZE, int(ty)*TiffTileWriter.TILE_SIZE)


if __name__ == '__main__':
    wmLvl = 25
    # wmLvl = 8
    x,y = (1<<(wmLvl-1)), (1<<(wmLvl-1))
    wmTlbr = (x,y,x+1024,y+1*1024)

    # tw = TiffTileWriter('/tmp/tst.tif', c=3, w=1024, h=1024, wmLvl, wmTlbr)
    tw = TiffTileWriter('/tmp/tst.tif', 3, wmLvl, wmTlbr)
    a = np.zeros((256,256,3), dtype=np.uint8)
    a[50:60, 50:60] = 200
    tw.writeTile(0,0,a)
    tw.writeTile(1,1,a)
