
from gtsam import (Cal3_S2, DoglegOptimizer,
                         GenericProjectionFactorCal3_S2, Marginals,
                         NonlinearFactorGraph, PinholeCameraCal3_S2, Point3,
                         Pose3, PriorFactorPoint3, PriorFactorPose3, Rot3, Values)

def show_matches(imgsa, imgsb, ptsa, ptsb, sigmas):
    pass

class Solver:
    def __init__(self):
        pass


    # NOTE: This may be called multiple times. This is preferred to calling it once and caching images, because
    #       that'd require potentially lots of RAM for storing decoded images.
    def get_loader(self, loadImages=True):
        from loading.terrapixel import TerraPixelLoader
        tpl = TerraPixelLoader('/data/inertialLabs/flightFeb15/irnOutput/1707947224/eval.bin', loadImages=loadImages)
        return tpl

    def get_matches(self):
        hist = []

        lookback = [1,2,4,8,16]

        # Oriented backwards (i -> j)
        allMatches = {}
        allFramesNoImages = {}

        for i,fwpp in enumerate(tpl):
            allMatches[i] = {}

            for bi in lookback:

                if len(hist)-bi >= 0:
                    matches = try_match(fwpp, hist[-bi])
                    if matches is None:
                        pass
                    else:
                        ptsa,ptsb,sigma = matches

                        j = i - bi
                        allMatches[i][j] = ptsa,ptsb,sigma

            hist.append(fwpp)
            if len(hist) > lookback[-1]: hist = hist[-lookback[-1]:]
            allFramesNoImages[i] = fwpp.dropImage()

        return allMatches, allFramesNoImages

    def run(self):
        matches, frames = self.get_matches()

        #
        # Create a pose state for each frame
        # Create a pose prior factor for each pose state
        #
        # Create a "landmark" position state for each unique observed keypoint
        # Create a factor for each observed keypoint + frame
        #
        # Create a factor for each landmark enforcing it to lie on DTED at it's current horizontal position.
        # NOTE: This ought to be done at __each iteration anew__.
        #       But that may require custom factors (can this be done in python?)
        #       So at first, maybe just apply elevation constraint according to initial/prior projected position.
        #

        for k,v in matches.items():
            fwpp = frames[k]
            K = Cal3_S2(*fwpp.frame.intrin.f, 0, *fwpp.frame.intrin.c)



    # Try to match two frames.
    def try_match(self, fa,fb):
        ptsa,ptsb, conf, sigma = self.matcher(fa,fb)
        nvalid = (conf>.5).long().sum()
        if nvalid < 5:
            print(f' - too few good matches ({nvalid} / 5)')
            return None
        return ptas,ptsb,sigma

if __name__ == '__main__':
    s = Solver()
    s.run()
