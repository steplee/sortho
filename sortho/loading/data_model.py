import numpy as np, cv2, os
from copy import deepcopy

'''
The only types needed for code shared amongst the `sortho/*` modules and with external inputs.
'''

class Intrin:
    def __init__(self, wh, f, c, d):
        self.wh = wh
        self.f,self.c,self.d = f,c,d

    def __repr__(self):
        return f'wh: {self.wh} f: {self.f} c: {self.c} d: {self.d}'

# pq is wrt LTP (world-from-body, RWF)
class PoseEcef:
    def __init__(self, pos, pq):
        self.pos = pos
        self.pq = pq

    def __repr__(self):
        return f'pos: {self.pos}, pq {self.pq}'

class Frame:
    def __init__(self, tstamp, img, intrin, sq):
        self.tstamp = tstamp
        self.img = img
        self.intrin = intrin
        self.sq = sq

    def dropImage(self):
        it = deepcopy(self)
        it.img = None
        return it

class FrameWithPosePrior:
    def __init__(self, frame, posePrior, posePriorSigmas):
        self.frame = frame
        self.posePrior = posePrior
        self.posePriorSigmas = posePriorSigmas

    def dropImage(self):
        it = deepcopy(self)
        it.frame.img = None
        return it

