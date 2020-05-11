
import numpy as np
import scipy as sp
import scipy.ndimage

import logging
logger = logging.getLogger(__name__)


def optional(x,y,func):
    if x is None:
        return func(y)
    else:
        rez=np.require(x,dtype=np.float64)
        assert rez.shape==tuple(y)
        return rez

def optional_const(x,y):
    if x is None:
        return y
    else:
        rez=np.require(x,dtype=np.float64)
        assert rez.shape==y.shape
        return rez


def optional_eye(x,y):
    if x is None:
        return np.eye(y)
    else:
        rez=np.require(x,dtype=np.float64)
        assert rez.shape==(y,y)
        return rez

def kill_whitespace(s):
    return re.sub("\s+",s)

def phasing(B,q):
    R,C,J=B.shape
    Z = np.zeros((R,C,J))
    Z[0] = B[0]
    for r in range(1, R):
        Z[r] = q[:, None]*Z[r-1] + B[r]
    return Z

def quadratic_form_to_nnls_form(Gamma,phi,lo=1e-10):
    A=sp.linalg.cholesky(Gamma+np.eye(Gamma.shape[0])*lo,lower=False)
    b=np.linalg.solve(A.T,phi)
    return A,b

def simple_downsample(X,axis):
    if X.shape[0]%2==0:
        return X[::2] + X[1::2]
    else:
        return X[:-1:2] + X[1::2]

def nonnegative_update(apply_Gamma,phi,y,lo=0,backtracking=.9,maxiter=10):
    '''
    Consider the problem of optimizing

        .5* y^T Gamma y - phi^T y

    subject to the constraint that (y>=lo).all()

    given an initial guess for y,
    this function finds a new value for y,
    which is gauranteed to be Not Worse!
    '''

    gY=apply_Gamma(y)

    # get a search direction
    search_dir = phi - gY
    assert not np.isnan(search_dir).any()

    # but we shouldn't go negative if we have active constraints!
    bad_id = (y <= lo)*(search_dir <= 0)
    search_dir[bad_id] = 0

    # assuming there are at least some nontrivial directions, do something
    if np.sum(search_dir**2)>1e-10:
        # get old loss
        old_loss = .5*np.sum(y*gY) - np.sum(phi*y)

        # how far should we go in this modified search direction?
        Gamma_search_dir=apply_Gamma(search_dir)
        bunsi = np.sum(phi*search_dir) - np.sum(y*Gamma_search_dir)
        bunmu = np.sum(search_dir * Gamma_search_dir)
        lr= bunsi/bunmu
        assert not np.isnan(lr)

        proposed_y = np.clip(y+lr*search_dir,lo,None)
        new_loss = .5*np.sum(proposed_y*apply_Gamma(proposed_y)) - np.sum(phi*proposed_y)

        # backtrack as necessary
        for i in range(maxiter):
            if new_loss<old_loss: # we made progress!  done!
                return proposed_y
            else:  # we didn't.  lets try to backtrack a bit
                logger.info('backtracking')
                lr=lr*backtracking
                proposed_y = np.clip(y+lr*search_dir,lo,None)
                new_loss = .5*np.sum(proposed_y*apply_Gamma(proposed_y)) - np.sum(phi*proposed_y)

        # even after backtracking a bunch of times
        # we still seem to be just making things worse.
        # abort!
        return y
    else:
        return y
