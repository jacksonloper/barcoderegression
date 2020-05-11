from . import denselearner
from . import helpers
import numpy as np
import scipy as sp
import scipy.ndimage
import numpy.random as npr

from . import blurkernels

def unif(a,b,*shape):
    return npr.rand(*shape)*(b-a)+a

def simulate(codebook,spatial_dims,num_polonies,
            blur=3,noise=.1,genedistr=None,
            varphi=None,scale_lo=1,scale_hi=2,rho=None,
            alpha=None):
    R,C,J=codebook.shape
    spatial_dims=tuple(spatial_dims)

    # get rolony densities
    if genedistr is None:
        genedistr=np.ones(J)/J

    M=np.prod(spatial_dims)
    positions = npr.randint(0,M,size=num_polonies)
    genes = npr.choice(J,p=genedistr,size=num_polonies)  # genes[i] ~ Categorical(genedistr)
    F=np.zeros((M,J))
    F[positions,genes]=unif(scale_lo,scale_hi,num_polonies)
    F=F.reshape(spatial_dims+(J,))

    # make model
    model = denselearner.Model(codebook,spatial_dims,blur_level=blur,F=F,varphi=varphi,rho=rho,alpha=alpha)

    # get obs
    X_without_noise=model.reconstruction()
    X_with_noise = X_without_noise + npr.rand(*X_without_noise.shape)*noise

    return dict(model=model,X=X_with_noise.numpy(),X_without_noise=X_without_noise.numpy())
