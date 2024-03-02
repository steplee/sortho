import numpy as np, torch
import pickle
from gtsam import (Cal3_S2, DoglegOptimizer,
                    GenericProjectionFactorCal3_S2, Marginals,
                    NonlinearFactorGraph, PinholeCameraCal3_S2, Point3,
                    ISAM2, 
                    PriorFactorCal3_S2, noiseModel, GeneralSFMFactor2Cal3_S2,
                    LevenbergMarquardtParams, LevenbergMarquardtOptimizer,
                    Pose3, PriorFactorPoint3, PriorFactorPose3, Rot3, Values)

from gtsam import symbol_shorthand
L = symbol_shorthand.L
X = symbol_shorthand.X
B = symbol_shorthand.B
K = symbol_shorthand.K

from sortho.loading.data_model import FrameWithPosePrior, PoseEcef
from sortho.matching.loftr import LoftrMatcher
from sortho.utils.etc import get_ltp, rodrigues, to_torch
from sortho.geometry.ray_march_dted import DtedRayMarcher

assert False, 'even with 3 images -- huge changes -- are intrinsics messed up?'

# def show_matches(imgsa, imgsb, ptsa, ptsb, sigmas): pass

def make_gtsam_pose(pose, sq=None, makeEcef=True, origin=None):
    from sortho.utils.q import q_to_matrix
    R = q_to_matrix(pose.pq)
    if makeEcef:
        Ltp = get_ltp(pose.pos[None])[0]
        R = Ltp @ R

    if sq is not None:
        R = R @ q_to_matrix(sq)

    p = (pose.pos-origin) if origin is not None else pose.pos

    rot = Rot3(R)
    return Pose3(rot, p)

def unmake_gtsam_pose(pose3, sq=None, unmakeEcef=True, origin=None):
    from sortho.utils.q import matrix_to_q, q_to_matrix

    p = pose3.translation()
    if origin is not None:
        p = p + origin

    R = pose3.rotation().matrix()

    if sq is not None:
        R = R @ q_to_matrix(sq).T

    if unmakeEcef:
        Ltp = get_ltp(p[None])[0]
        R = Ltp.T @ R

    return PoseEcef(p, matrix_to_q(R))


# Quantize XY keypoints and aggregate them with consistent IDs amongst multiple frames
# FIXME: Record dict of observations and _SEPERATE_ dict of keypoint locations, prevents need for quantization.
def aggregrate_keypoints_0(matchesIJ):

    # For each frame, store a dict of quantized XY keypoint to list of 
    frameKptss = {}

    for i, matchesJ in matchesIJ.items():
        for j, matches in matchesJ.items():
            ptsi, ptsj, conf = matches
            if isinstance(ptsi, torch.Tensor): ptsi = ptsi.cpu().numpy()
            if isinstance(ptsj, torch.Tensor): ptsj = ptsj.cpu().numpy()
            if isinstance(conf, torch.Tensor): conf = conf.cpu().numpy()
            ptsi = ptsi.astype(int).tolist()
            ptsj = ptsj.astype(int).tolist()

            if i not in frameKptss: frameKptss[i] = {}
            if j not in frameKptss: frameKptss[j] = {}
            frameKptsI = frameKptss[i]
            frameKptsJ = frameKptss[j]
            for I in range(len(ptsi)):
                pti, ptj, cnf = tuple(ptsi[I]), tuple(ptsj[I]), conf[I]
                if pti not in frameKptsI: frameKptsI[pti] = [(j, ptj, cnf)]
                else:                     frameKptsI[pti].append((j, ptj, cnf))
                if ptj not in frameKptsJ: frameKptsJ[ptj] = [(i, pti, cnf)]
                else:                     frameKptsJ[ptj].append((i, pti, cnf))

    return frameKptss

#
# TODO: as a work around to the multiple 'seenBy' observing keypoints in the same keyframe,
#       maybe multiple the sigma by 2^num_observers_from_same_frame!
# I don't see a great way to fix the issue. Maybe just disregard all such estimates too?
#
def aggregrate_keypoints(matchesIJ):
    observations = {} # dict[frameKey => dict[quantizedPt => landmarkIdx]]
    landmarks = []

    for i, matchesJ in matchesIJ.items():
        for j, matches in matchesJ.items():
        # for j, matches in list(matchesJ.items())[::-1]:
            ptsi, ptsj, sgma = matches
            if isinstance(ptsi, torch.Tensor): ptsi = ptsi.cpu().numpy()
            if isinstance(ptsj, torch.Tensor): ptsj = ptsj.cpu().numpy()
            if isinstance(sgma, torch.Tensor): sgma = sgma.cpu().numpy()
            ptsi = ptsi.astype(int).tolist()
            ptsj = ptsj.astype(int).tolist()

            if i not in observations: observations[i] = {}
            if j not in observations: observations[j] = {}

            for I in range(len(ptsi)):
                kj = j, *ptsj[I]
                ki = i, *ptsi[I]
                assert j < i

                if kj not in observations[j]:
                    lid = len(landmarks) # lid = landmarkIdx
                    # landmarks.append(dict(seenBy=[kj]))
                    landmarks.append(dict(firstSeenBy=kj, seenBy=set([kj])))
                    observations[j][kj] = lid, sgma[I]
                else:
                    lid, _ = observations[j][kj]

                # WARNING: This is can be re-assigned multiple times -- `ki` can be repeated for multiple of the `matchesJ` loop.
                #          This means that a landmark can have MULTIPLE observing keypoints in a SINGLE image, which is a little weird, but not an error?
                observations[i][ki] = lid, sgma[I]
                # landmarks[lid]['seenBy'].append(ki)
                landmarks[lid]['seenBy'].add(ki)

    # print('Observations:\n', observations)
    # print('Landmarks:\n', landmarks)
    # print('Landmarks[0:20]:\n', landmarks[0:20])

    if 1:
        n_seen = sum(len(l['seenBy']) for l in landmarks)
        seen_set = set()
        for l in landmarks:
            for p in l['seenBy']:
                seen_set.add(p)
        n_seen_uniq = len(seen_set)
        n_pts  = 0
        for i, matchesJ in matchesIJ.items():
            for j, matches in matchesJ.items():
                n_pts += len(matches[0]) * 2
        print(n_seen,n_seen_uniq,n_pts)
        lsizes = [len(l['seenBy']) for l in landmarks]
        print(' - Landmark SeenBy Histogram')
        for i,bin in enumerate(np.histogram(lsizes, bins=max(lsizes), range=(0,max(lsizes)))[0]):
            if i > 1:
                print('   ',i,bin)

    return observations, landmarks






class Solver:
    def __init__(self, conf):
        self.conf = conf
        self.matcher = LoftrMatcher(**conf.matcher)
        self.dtedRayMarcher = DtedRayMarcher(**conf.dtedRayMarcher)
        self.show = conf.show


    def get_loader(self, loadImages=True):
        from sortho.loading.random_access import RandomAccessLoader
        loader = RandomAccessLoader(**self.conf.input, loadImages=loadImages)
        return loader

    def get_matches(self):
        hist = [] # Store recent fwpp in a deque

        # lookback = [1,2,4,8,16]
        # lookback = [1,2,3,5,8]
        # lookback = [1,2,3,5]
        lookback = [1,2]

        # Oriented backwards (i -> j)
        allMatchesIJ = {}
        allFramesNoImages = {}

        tpl = self.get_loader(loadImages=True)
        for i,fwpp in enumerate(tpl):
            allMatchesIJ[i] = {}

            for bi in lookback:

                if len(hist)-bi >= 0:
                    # print('COMPARE', len(allMatchesIJ)-1, len(allMatchesIJ)-bi-1)
                    matches = self.try_match(fwpp, hist[-bi])
                    if matches is None:
                        pass
                    else:
                        ptsa,ptsb,sigma = matches

                        j = i - bi
                        allMatchesIJ[i][j] = ptsa,ptsb,sigma

            hist.append(fwpp)
            if len(hist) > lookback[-1]: hist = hist[-lookback[-1]:]
            allFramesNoImages[i] = fwpp.dropImage()

        return allMatchesIJ, allFramesNoImages

    def run(self):
        matchesIJ, frames = self.get_matches()

        '''
        # Copy and transpose map
        matchesJI = {}
        for i,matchesJ in matchesIJ.items():
            for j,matches in matchesJ.items():
                if j not in matchesJI: matchesJI[j] = {}
                matchesJI[j][i] = matches

        self.run_nlls(matchesIJ, matchesJI, frames)
        '''

        frameObss, landmarks = aggregrate_keypoints(matchesIJ)
        self.run_nlls(frameObss, landmarks, frames)

    #
    # NOTE: I am _NOT_ using ISAM2 here, but just a NLLS BA setup
    #
    # def run_nlls(self, matchesIJ, matchesJI, frames):
    def run_nlls(self, frameObss, landmarks, frames):

        # framObs :: dict[frameKey => dict[quantizedPt => landmarkIdx]]

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
        # NOTE: I cannot find out how to create a node from the transformation of two others. This is unbelievable!
        #       It looks like the C++ `ReferenceFrameFactor` is needed, but not available in python.
        #       That means I cannot have a shared bias pose that all others are transformed by.
        #       That means I also have to apply the sensor_quaternion to the main pose itself (rather than splitting into two)
        #

        print(' - Building Factor Graph')
        Kdict = {}

        initial = Values()
        graph = NonlinearFactorGraph()

        # pose2frameKey = {}
        poseKeys = []
        # frameKeyAndPixelToLandmarkId = {}
        # landmarkIdToFrameKeyAndPixel = {}

        # TODO: Lower the Z sigma on the last few iters
        # landmark_prior_noise = noiseModel.Isotropic.Sigmas(np.array((500,500, 10.)))
        landmark_prior_noise = noiseModel.Isotropic.Sigmas(np.array((9500,9500, 10.)))

        posePriorFactors = []
        landmarkPriorFactors = []
        projFactors = []

        origin = list(frames.values())[0].posePrior.pos

        # for k in matchesIJ.keys():
        for k in frameObss.keys():
            fwpp = frames[k]

            # Get or create camera model node
            # NOTE: Should use PinholeCamera node, but I'll handle pose+camera myself to learn.
            if 1:
                fx,fy = fwpp.frame.intrin.f.astype(int)
                fkey = fx*10_000 + fy
                Knew = False
                if fkey not in Kdict: Kdict[fkey], Knew = (len(Kdict),Cal3_S2(*fwpp.frame.intrin.f, 0, *fwpp.frame.intrin.c)), True
                Kid, KK = Kdict[fkey]

                # If created new camera model, add prior on it.
                if Knew:
                    cam_prior_noise = np.array((30,30, 1e-5, 10,10.))
                    graph.add(PriorFactorCal3_S2(K(Kid), KK, noiseModel.Isotropic.Sigmas(cam_prior_noise)))
                    initial.insert(K(Kid), KK)

            # Create pose prior
            if 1:
                pp = fwpp.posePrior
                pp_sigmas = fwpp.posePriorSigmas
                # pp_sigmas *= 10 # WARNING:
                print('pp_sigma', pp_sigmas)
                # Xid = len(pose2frameKey)
                # pose2frameKey[Xid] = k
                Xid = k
                poseKeys.append(Xid)
                XX = make_gtsam_pose(pp, makeEcef=True, sq=fwpp.frame.sq, origin=origin)
                posePriorFactors.append(PriorFactorPose3(X(Xid), XX, noiseModel.Isotropic.Sigmas(pp_sigmas)))
                graph.add(posePriorFactors[-1])
                initial.insert(X(Xid), XX)

            # FIXME: Graph is wrong, leads to zero projection errors. logical error must fix.

            # Create landmark observation factors. If any landmarks are unseen add and add initialize them. Add elevation constraint on vertical axis.
            if 1:
                # Combine forward and backward looking matches
                # matches = [list(matchesIJ.items()) + list(matchesJI.items())]
                frameObs = frameObss[k]
                pts = np.array(list(frameObs.keys()))[:,1:] # IXY -> XY
                landmarkPts0 = self.intersect_terrain(XX, KK, pts, origin)
                # camera = PinholeCameraCal3_S2(X(Xid), K)

                # for ii, (ipt, (jpts)) in enumerate(frameObs.items()):
                for ii, (ipt, (Lid, sigma)) in enumerate(frameObs.items()):

                    if landmarks[Lid]['firstSeenBy'] == ipt:
                        # Not efficient.
                        LL = Point3(landmarkPts0[ii])
                        landmarkPriorFactors.append(PriorFactorPoint3(L(Lid), LL, landmark_prior_noise))
                        graph.add(landmarkPriorFactors[-1])
                        initial.insert(L(Lid), LL)

                    # noise = noiseModel.Robust.Create(noiseModel.mEstimator.Huber.Create(20), noiseModel.Isotropic.Sigma(2,14))
                    noise = noiseModel.Robust.Create(noiseModel.mEstimator.Huber.Create(15), noiseModel.Isotropic.Sigma(2,sigma))
                    # print(sigma)
                    # noise = noiseModel.Isotropic.Sigma(2,10) # FIXME: use sigma
                    # noise = noiseModel.Isotropic.Sigma(2,.00001) # FIXME: use sigma
                    # noise = noiseModel.Isotropic.Sigma(2,14) # FIXME: use sigma

                    # graph.add(GenericProjectionFactorCal3_S2(ipt, noise, X(i), L(Lid), Kid))
                    # ipt = (ipt[0]+90, ipt[1])
                    projFactors.append(GeneralSFMFactor2Cal3_S2(pts[ii], noise, X(Xid), L(Lid), K(Kid)))
                    graph.add(projFactors[-1])


        # NOTE: Because the landmark terrain elevation constraints change according to current state,
        #       We should not run for many iterations. We have to keep updating the constraints as the state values change.
        # params = LevenbergMarquardtParams()
        print(' - Creating Optimizer')
        # params = LevenbergMarquardtParams.CeresDefaults()
        params = LevenbergMarquardtParams()
        params.setMaxIterations(100)
        params.setVerbosityLM('DAMPED')
        optimizer = LevenbergMarquardtOptimizer(graph, initial, params)
        print(' - Optimizing')
        final = optimizer.optimize()

        # FIXME: Do this multiple times -- updating the terrain elevation constraint each time!

        def sumErr(factors, values):
            mse = 0
            for factor in factors: mse += factor.error(values)
            return np.sqrt(mse/len(factors))
        def printErrs(values):
            print(f'\t- sum total    : {graph.error(values):7.9f}')
            print(f'\t- posePrior    : {sumErr(posePriorFactors, values):7.9f}')
            print(f'\t- landmarkPrior: {sumErr(landmarkPriorFactors, values):7.9f}')
            print(f'\t- projection   : {sumErr(projFactors, values):7.9f}')
        print(' - Initial RMSEs')
        printErrs(initial)
        print(' - Final RMSEs')
        printErrs(final)

        ediff, ldiff = 0, 0
        for Xid in poseKeys:
            a = initial.atPose3(X(Xid)).translation()
            b = final  .atPose3(X(Xid)).translation()
            ediff += np.linalg.norm(a-b)
        for Lid in range(len(landmarks)):
            a = initial.atPoint3(L(Lid))
            b = final  .atPoint3(L(Lid))
            ldiff += np.linalg.norm(a-b)
        print(f' - Differences')
        print(f'\t- landmarks: {ldiff/len(landmarks):7.9f}')
        print(f'\t- poseEyes : {ediff/len(landmarks):7.9f}')

        self.write_output(final, graph, poseKeys, origin)

        if self.show:
            self.make_gtsam_viz('initial', origin, [initial,final], frames, poseKeys, landmarks)

    def make_gtsam_viz(self, name, origin, valuess, frames, poseKeys, landmarks):
        import matplotlib.pyplot as plt

        # Get Ltp matrix
        poss = []
        for frame in frames.values():
            poss.append(frame.posePrior.pos)
        pos0 = np.stack(poss,0).mean(0)
        Ltp = get_ltp(pos0[None])[0]

        for label, values in zip(('initial', 'final'), (valuess[0], valuess[-1])):

            # Get all optical centers
            octrs = []
            for Xid in poseKeys:
                XX = values.atPose3(X(Xid))
                p = XX.translation() + origin
                print(label,Xid,XX.translation(), p)
                p_local = Ltp.T @ p
                octrs.append(p_local)
            octrs = np.stack(octrs,0)

            # Get all landmark pts
            lpts = []
            for Lid in range(len(landmarks)):
                LL = values.atPoint3(L(Lid)) + origin
                p = LL
                p_local = Ltp.T @ p
                lpts.append(p_local)
            lpts = np.stack(lpts,0)

            plt.scatter(octrs[:,0], octrs[:,1], label=label+'_pose')
            plt.scatter(lpts[:,0], lpts[:,1], label=label+'_lpt', s=.2)
        plt.legend()
        plt.show()





    def intersect_terrain(self, gtsamPose, gtsamIntrin, pixPts, origin):
        eye   = gtsamPose.translation() + origin
        R     = gtsamPose.rotation().matrix()
        cam_f = gtsamIntrin.vector()[0:2]
        cam_c = gtsamIntrin.vector()[3:5]

        print(pixPts)
        fpts = (pixPts - cam_c) / cam_f

        eye, R, fpts = to_torch(eye, R, fpts)

        rpts = self.dtedRayMarcher.march(eye, R, fpts, R_is_ecef=True).cpu().numpy()
        depths = np.linalg.norm(rpts - eye.cpu().numpy()[None], axis=1).mean()
        print('avg depths', depths)
        return rpts - origin



    # Try to match two frames.
    def try_match(self, fa,fb):
        matches = self.matcher.match(fa.frame.img,fb.frame.img, debugShow=dict(wait=1))
        ptsa, ptsb, conf, sigma = matches['apts'], matches['bpts'], matches['conf'], matches['sigma']
        ptsa, ptsb, conf, sigma = (t.cpu() for t in (ptsa,ptsb,conf,sigma))
        nvalid = (conf>.5).long().sum()
        if nvalid < 5:
            print(f' - too few good matches ({nvalid} / 5)')
            return None
        return ptsa,ptsb,sigma

    def write_output(self, final, graph, poseKeys, origin):
        meta = {}
        meta['imageCompressionExt'] = '.jpg'
        fwpps = []

        marginals = Marginals(graph, final)

        for i,fwpp0 in enumerate(self.get_loader(False)):
            # Xid = frameKey2poseKey[fwpp0.frame.tstamp]
            Xid = i
            pp = unmake_gtsam_pose(final.atPose3(X(Xid)), sq=fwpp0.frame.sq, origin=origin)
            print(fwpp0.posePrior, '->', pp)
            pp_sigmas = np.diagonal(marginals.marginalCovariance(X(Xid)))
            fwpps.append(FrameWithPosePrior(fwpp0.frame, pp, pp_sigmas))

        meta['framesWithPosePriors'] = list(fwpps)
        with open(self.conf.outputPath,'wb') as fp:
            pickle.dump(meta, fp)
        print(f' - Wrote \'{self.conf.outputPath}\'')


if __name__ == '__main__':
    from omegaconf import OmegaConf
    conf = OmegaConf.create({
        'matcher': {},
        'input': {},
        # 'input': dict(path='/data/inertialLabs/flightFeb15/sortho.ra', frameStride=8, maxFrames=25),
        # 'input': dict(path='/data/inertialLabs/flightFeb15/sortho.ra', frameStride=8, maxFrames=8),
        # 'input': dict(path='/data/inertialLabs/flightFeb15/sortho.ra', frameStride=8, maxFrames=5),
        'dtedRayMarcher': dict(dtedPath='/data/elevation/srtm/srtm.fft', iters=25),
        'outputPath': '/data/inertialLabs/flightFeb15/sortho.opt.ra',
        'show': False,
        # 'solver': dict(huber=True),
    })
    conf1 = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf,conf1)
    s = Solver(conf)
    s.run()
