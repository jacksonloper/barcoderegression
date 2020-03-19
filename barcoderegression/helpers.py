
import numpy as np
import scipy as sp
import scipy.ndimage

def optional(x,y,nm=''):
    if x is None:
        return y
    else:
        x=np.require(x)
        if x.shape!=y.shape:
            raise ValueError(f"shape for parameter {nm} should be {y.shape}, not {x.shape}")
        return x

def clip(x,lo=0):
    return np.clip(x,lo,None)

def phasing(B,q):
    R,C,J=B.shape
    Z = np.zeros((R,C,J))
    Z[0] = B[0]
    for r in range(1, R):
        Z[r] = q[:, None]*Z[r-1] + B[r]
    return Z


def loss_at_q(X,spinfo,barcode,lam):
    X=X.reshape((spinfo.M,barcode.N))
    Xmab = X - spinfo.arav[:,None] - barcode.brav[None,:]

    def laq(q):
        Z=phasing(barcode.B,q)
        Gtilde = np.einsum('ck, rkj -> rkj',barcode.g,Z)
        G= Gtilde*barcode.alpha[:,:,None]
        Grav = G.reshape((barcode.N,barcode.J))

        reconstruction_loss = np.sum((Xmab - spinfo.Frav@Grav.T)**2)
        l1_loss = np.sum(spinfo.Frav @ Grav.T)

        return .5*reconstruction_loss + .5*lam*l1_loss

    return laq


def loss(X,spinfo,barcode,lam,eps):
    '''
    Input:
    - X -- (dims[0],dims[1] ..., dims[-1], number of rounds, number of channels)
    - spinfo -- SpatialInfo object
    - barcode -- Barcode object
    - lam -- scalar
    - eps -- scalar

    Evaluates the the objective

      L = .5*|X - F G^T - a1 - b1|^2_2 + .5*lam |FG^T|_1 + .5*eps |Y|^2

    Returns a dictionary indicating various features of this loss:
    - the "reconstruction", i.e. .5*|X - F G^T - a1 - b1|^2_2
    - the "l1", i.e. .5*|FG^T|_1
    - the "l2", i.e. .5*|Y|^2
    - "overall" -- the whole loss put together
    '''

    X=X.reshape((spinfo.M,barcode.N))
    FGt=spinfo.Frav@barcode.Grav.T

    resid = X - FGt - spinfo.arav[:,None] - barcode.brav[None,:]

    lossinfo= dict(
        reconstruction = .5*np.sum(resid**2),
        l1 = .5*np.sum(FGt),
        l2 = .5*np.sum(spinfo.Y**2),
        lam=lam,
        eps=eps
    )

    lossinfo['overall']=lossinfo['reconstruction'] + lam*lossinfo['l1'] + eps*lossinfo['l2']
    return lossinfo


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
