from ..data_model import *
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

class TerraPixelLoader():
    def __init__(self, path, frameStride=1, loadImages=True, maxFrames=0):
        super().__init__()

        self.reader = Reader(path, lazy=True)
        self.lastAsd = None
        self.loadImages = loadImages
        self.frameStride = frameStride
        self.maxFrames = maxFrames
        self.nframesIn = 1
        self.nframesOut = 1

    def iterFramesWithPosePriors(self):
        for Ty, buf, sz in self.reader:

            if self.maxFrames > 0 and self.nframesOut > self.maxFrames: return

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

                self.nframesIn += 1
                if self.nframesIn % self.frameStride == 0:
                    self.nframesOut += 1

                    # Get frame
                    jsf = Ty()
                    jsf.Init(buf, sz)
                    img = cv2.imdecode(jsf.JpegDataAsNumpy(), cv2.IMREAD_COLOR) if self.loadImages else jsf.JpegDataAsNumpy()
                    frame = Frame(jsf.Tstamp(), img, createIntrinFromCamModel(jsf.Cam()), decodeQuat(self.lastAsd.Sq))

                    # Get posePrior
                    posWgs = self.lastAsd.PosAsNumpy()[None]
                    posEcef = transform_points_epsg(4326, 4978, posWgs)[0]
                    pq_rwf = decodeQuat(self.lastAsd.Pq)
                    pq_enu = q_enu_from_rwf(pq_rwf)
                    pp = PoseEcef(posEcef, pq_enu)
                    pos_sigma = np.full((70,), 3)
                    pos_sigma = self.lastAsd.PosSigma()
                    ori_sigma = np.full((3,),.004)
                    pp_sigmas = np.concatenate((ori_sigma, pos_sigma,))

                    yield FrameWithPosePrior(frame, pp, pp_sigmas)


if __name__ == '__main__':

    #
    # Convert to "random access" format.
    #

    import pickle
    from omegaconf import OmegaConf
    conf = OmegaConf.from_cli()
    with open(conf.output, 'wb') as fp:
        del conf.output
        tpl = TerraPixelLoader(**conf, loadImages=False)
        meta = {}
        meta['imageCompressionExt'] = '.jpg'
        meta['framesWithPosePriors'] = list(tpl.iterFramesWithPosePriors())
        pickle.dump(meta, fp)
        '''
        for fwpp in tpl.framesWithPosePriors():
            frameKey = fwpp.frame.tstamp
            jpegData = fwpp.frame.img#.tobytes()
        '''
