import numpy as np, cv2
import os, sys
try:
    import gdal
except:
    from osgeo import gdal, gdalconst

'''
Note: This will be very slow without overviews.
'''

class TiffViewer:
    def __init__(self, path):
        self.path = path
        self.dset = gdal.Open(path, gdal.GA_ReadOnly)
        self.H = self.dset.RasterYSize
        self.W = self.dset.RasterXSize
        self.aspect_wh = self.W/self.H

        self.tlbr = np.array((0,0,self.W-1,self.H-1),dtype=np.float32)
        self.update()

    def update(self):
        tlbr = self.tlbr

        a = self.aspect_wh
        if a > 1:
            s = 1800
            w = s
            h = int(s/a+.5)
        else:
            s = 1024
            h = s
            w = int(s*a+.5)

        r = gdalconst.GRIORA_Bilinear
        xywh = np.concatenate((tlbr[:2], tlbr[2:]-tlbr[:2]))
        x,y,ww,hh = [int(a) for a in xywh]
        img = self.dset.ReadAsArray(x,y,ww,hh, buf_xsize=w,buf_ysize=h, resample_alg=r)
        if img is None:
            print('bad tlbr', tlbr)
            k = chr(cv2.waitKey(0))
            return False, k
        else:
            img = img.transpose(1,2,0)[...,:3]
            if img.shape[-1] == 3: img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)


            cv2.imshow(self.path, img)
            return True, chr(cv2.waitKey(0))

    def run(self):
        zoom_speed = .1
        move_speed = .1

        while True:

            good, key = self.update()
            if key == 'q': break

            tlbr0 = self.tlbr
            tlbr  = self.tlbr

            if key == 'w':
                m = tlbr.reshape(2,2).mean(0)
                d = (tlbr[2:] - tlbr[:2]) * .5 * (1-zoom_speed)
                tlbr = np.concatenate((m-d,m+d),0)

            if key == 's':
                m = tlbr.reshape(2,2).mean(0)
                d = (tlbr[2:] - tlbr[:2]) * .5 / (1-zoom_speed)
                tlbr = np.concatenate((m-d,m+d),0)

            if key == 'a' or key == 'd':
                d = (tlbr[2:] - tlbr[:2])[0] * .5 * (move_speed)
                d = np.array((d,0,d,0)) * (1 if key == 'd' else -1)
                tlbr += d

            if key == 'z' or key == 'c':
                d = (tlbr[2:] - tlbr[:2])[1] * .5 * (move_speed)
                d = np.array((0,d,0,d)) * (1 if key == 'z' else -1)
                tlbr += d

            # if good: self.tlbr = tlbr
            self.tlbr = tlbr



if __name__ == '__main__':
    tv = TiffViewer(sys.argv[1])
    tv.run()
