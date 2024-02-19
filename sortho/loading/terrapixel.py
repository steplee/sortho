from .base import *
from sortho.utils.geo import transform_points_epsg

# Unbelievably, flatbuffers generated python code uses a non-relative import, and requires adding the generated package directly to path.
# It cannot be tucked away inside an outer package like I wanted it to be..
import pyirn
from pyirn.analysis.read_write import Reader, Asd, SourceFrame, JpegSourceFrame, CameraModel, Quatf, AsdSource
import sys
pyirn_path = os.path.dirname(pyirn.__file__)
sys.path.append(pyirn_path)

def decodeQuat(q_):
    q = Quatf()
    q_(q)
    return np.array((q.W(), q.X(), q.Y(), q.Z()))
def createIntrinFromCamModel(c):
    return Intrin(c.WhAsNumpy(), c.FAsNumpy(), c.CAsNumpy(), c.DAsNumpy())

# Recall that all terrapixel platform orientations are stored in "RWF" coordinates.
# I want `orthoRectify` coordinates in ENU, however.
from sortho.utils.q import q_mult, q_exp, q_to_matrix
enu_from_rwf_ = q_exp(np.array((-np.pi*.5,0,0)))
def q_enu_from_rwf(a):
    return q_mult(enu_from_rwf_, a)
if 0:
    print(enu_from_rwf_)
    print(q_to_matrix(np.array((1,0,0,0))))
    print(q_to_matrix(q_enu_from_rwf(np.array((1,0,0,0)))))
    exit()

class TerraPixelLoader(BaseLoader):
    def __init__(self, path, loadImages=True):
        super().__init__()

        self.reader = Reader(path, lazy=True)
        self.lastAsd = None
        self.loadImages = loadImages

    def __iter__(self):
        self.reader_iter = iter(self.reader)
        return self

    def __next__(self):
        while True:
            Ty, buf, sz = next(self.reader_iter)

            if Ty == Asd:
                asd = Ty()
                asd.Init(buf, sz)

                src = asd.Source()
                if src == AsdSource.ePred and (self.lastAsd is None or asd.Htstamp() > self.lastAsd.Htstamp() + 5_000_000):
                    print(' - Warning: using Asd with "pred" source because self.lastAsd is null or 5 seconds old.')
                    self.lastAsd = asd
                elif src == AsdSource.eGnss:
                    self.lastAsd = asd

            if self.lastAsd is not None and Ty == JpegSourceFrame:

                # Get frame
                jsf = Ty()
                jsf.Init(buf, sz)
                img = cv2.imdecode(jsf.JpegDataAsNumpy(), cv2.IMREAD_COLOR) if self.loadImages else None
                frame = Frame(jsf.Tstamp(), img, createIntrinFromCamModel(jsf.Cam()), decodeQuat(self.lastAsd.Sq))

                # Get posePrior
                posWgs = self.lastAsd.PosAsNumpy()[None]
                posEcef = transform_points_epsg(4326, 4978, posWgs)[0]
                pq_rwf = decodeQuat(self.lastAsd.Pq)
                pq_enu = q_enu_from_rwf(pq_rwf)
                pp = PoseEcef(posEcef, pq_enu)
                pos_sigma = np.full((8,), 3)
                pos_sigma = self.lastAsd.PosSigma()
                print('pos_sigma', pos_sigma)
                pp_sigmas = np.concatenate((pos_sigma, np.full((3,),.001)))

                return FrameWithPosePrior(frame, pp, pp_sigmas)


if __name__ == '__main__':

    # Quick test.

    tpl = TerraPixelLoader('/data/inertialLabs/flightFeb15/irnOutput/1707947224/eval.bin', loadImages=False)
    for item in tpl:
        print(item.posePrior, item.frame.intrin)

