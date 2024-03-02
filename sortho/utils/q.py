import numpy as np

def log_R(R):
    t = np.arccos((np.trace(R) - 1) * .5)
    d = np.linalg.norm((R[2,1]-R[1,2], R[0,1]-R[1,0], R[0,2]-R[2,0]))
    if d < 1e-12: return np.zeros(3)
    return np.array((R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1])) * t / d
def q_exp(r):
    l2 = r@r
    if l2 < 1e-15:
        return np.array((1,0,0,0.))
    l = np.sqrt(l2)
    # a = l * np.pi * .5
    a = l * .5
    c,s = np.cos(a), np.sin(a)
    # return np.array((0,1,0,0))
    return np.array((c,*((s/l)*r)))

def q_mult(p,q):
    a1,b1,c1,d1 = p
    a2,b2,c2,d2 = q
    return np.array((
        a1*a2 - b1*b2 - c1*c2 - d1*d2,
        a1*b2 + b1*a2 + c1*d2 - d1*c2,
        a1*c2 - b1*d2 + c1*a2 + d1*b2,
        a1*d2 + b1*c2 - c1*b2 + d1*a2))

def q_to_matrix(q):
    r,i,j,k = q[0], q[1], q[2], q[3]
    return np.array((
        1-2*(j*j+k*k), 2*(i*j-k*r), 2*(i*k+j*r),
        2*(i*j+k*r), 1-2*(i*i+k*k), 2*(j*k-i*r),
        2*(i*k-j*r), 2*(j*k+i*r), 1-2*(i*i+j*j))).reshape(3,3)

# FIXME: Any more accurate way to do this
def matrix_to_q(R):
    return q_exp(log_R(R))
