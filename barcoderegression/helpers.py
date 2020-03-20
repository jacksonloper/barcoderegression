
import numpy as np
import scipy as sp
import scipy.ndimage

def optional(x,y,nm=''):
    if x is None:
        return y()
    else:
        return np.require(x)

def clip(x,lo=0):
    return np.clip(x,lo,None)

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

def nonnegative_update(apply_Gamma,phi,y,lo=0,backtracking=.9,maxiter=10):
    '''
    Consider the problem of optimizing

        .5* y^T Gamma y - phi^T y

    subject to the constraint that (y>=lo).all()

    given an initial guess for y,
    this function finds a new value for y whose
    '''

    # get a search direction
    search_dir = phi - apply_Gamma(y)
    assert not np.isnan(search_dir).any()

    assert not np.isnan(search_dir).any()

    # but we shouldn't go negative if we have active constraints!
    bad_id = (y <= lo)*(search_dir <= 0)
    search_dir[bad_id] = 0

    # assuming there are at least some nontrivial directions, do something
    if np.sum(search_dir**2)>1e-10:
        # get old loss
        old_loss = .5*np.sum(y*apply_Gamma(y)) - np.sum(phi*y)

        # how far should we go in this modified search direction?
        Gamma_search_dir=apply_Gamma(search_dir)
        bunsi = np.sum(phi*search_dir) - np.sum(y*Gamma_search_dir)
        bunmu = np.sum(search_dir * Gamma_search_dir)
        lr= bunsi/bunmu
        assert not np.isnan(lr)

        proposed_y = clip(y+lr*search_dir,lo)
        new_loss = .5*np.sum(proposed_y*apply_Gamma(proposed_y)) - np.sum(phi*proposed_y)

        # backtrack as necessary
        for i in range(maxiter):
            if new_loss<old_loss: # we made progress!  done!
                return proposed_y
            else:  # we didn't.  lets try to backtrack a bit
                lr=lr*backtracking
                proposed_y = clip(y+lr*search_dir,lo)
                new_loss = .5*np.sum(proposed_y*apply_Gamma(proposed_y)) - np.sum(phi*proposed_y)

        # even after backtracking a bunch of times
        # we still seem to be just making things worse.
        # abort!
        return y
    else:
        return y
