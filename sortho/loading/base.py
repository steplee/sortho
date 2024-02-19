import numpy as np, cv2, os
from copy import deepcopy


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
    def __init__(self, frame, posePrior, posePrior_sigmas):
        self.frame = frame
        self.posePrior = posePrior
        self.posePrior_sigmas = posePrior_sigmas

    def dropImage(self):
        it = deepcopy(self)
        it.frame.img = None
        return it



class BaseLoader:
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError()
    def __getitem__(self, i):
        raise NotImplementedError()

    def __iter__(self,path):
        self.ii = 0
        return self

    def __next__(self):
        self.ii += 1
        if self.ii >= len(self): raise StopIteration()
        return self[self.ii-1]

