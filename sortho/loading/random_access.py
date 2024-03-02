import pickle, cv2, numpy as np
from copy import deepcopy
from functools import lru_cache

'''
The only class that should be used for data access.
Use any of the `sortho/loading/converters/*` code to ingest data into the expected format used here.
'''

class RandomAccessLoader:
    def __init__(self, path, loadImages=True, maxFrames=int(8e8), firstFrame=0, lastFrame=int(8e8), frameStride=1):
        self.path = path
        print(f" - Loading '{path}'")
        with open(path,'rb') as fp:
            self.meta = pickle.load(fp)
        self.imageCompressionExt = self.meta['imageCompressionExt']
        self.loadImages = loadImages

        fwpps = self.meta['framesWithPosePriors'][firstFrame:lastFrame][::frameStride][:maxFrames]

        self._framesWithPosePriors = fwpps
        # self.frameKeyToIndex = {f.frame.tstamp: i for i,f in enumerate(self._framesWithPosePriors)}

    def __len__(self):
        return len(self._framesWithPosePriors)

    @lru_cache(maxsize=64)
    def __getitem__(self, k):
        if not self.loadImages:
            # return self._framesWithPosePriors[self.frameKeyToIndex[k]]
            return self._framesWithPosePriors[k]
        else:
            fwpp = deepcopy(self._framesWithPosePriors[k])
            fwpp.frame.img = cv2.imdecode(np.array((fwpp.frame.img)), 0)
            if fwpp.frame.img.ndim == 2:
                fwpp.frame.img = fwpp.frame.img[...,None]
            return fwpp


    def iterFramesWithPosePriors(self):
        for i,fwpp in enumerate(self._framesWithPosePriors):
            yield self[i]

if __name__ == '__main__':

    from omegaconf import OmegaConf
    conf = OmegaConf.from_cli()
    for fwpp in RandomAccessLoader(conf.input).iterFramesWithPosePriors():
        print(fwpp.frame.img)
