from . import parameters
from . import helpers
import numpy as np
import scipy as sp
import scipy.ndimage
import numpy.random as npr

from . import blurkernels

def simulation_from_B(B,spatial_dims,num_spots,blursize,noise=.1,lo=1e-10,lam=.1,genedistr=None):
    spatial_dims=tuple(spatial_dims)
    R,C=B.shape[:2]
    J=B.shape[-1]
    dims=spatial_dims+(R,C)
    dims_F=spatial_dims+(J,)

    if genedistr is None:
        genedistr=np.ones(J)/J

    rho=np.zeros(C)
    alpha=npr.rand(R,C)+.2  # Uniform[.2,1.2]
    varphi=np.eye(C)

    # sample ground truth Fs
    M = np.prod(spatial_dims)
    positions = npr.randint(0,M,size=num_spots)
    genes = npr.choice(J,p=genedistr,size=num_spots)  # genes[i] ~ Categorical(genedistr)

    # make the spatial loadings
    F = np.zeros((M,J))
    F[positions,genes]=npr.rand(num_spots)*30+10  # brightness ~ Uniform[10,40]
    K=blurkernels.ContiguousBlur(spatial_dims,blursize)
    F_blurred =K@F

    # make a b
    a=np.zeros(spatial_dims).ravel()
    b=np.zeros((R,C))

    # get it
    model=parameters.Model(B,K,F=F.reshape((-1,J)),a=a,b=b,alpha=alpha,rho=rho,varphi=varphi,lo=lo,lam=lam)
    X_without_noise=model.reconstruction()  # FG^T + a1 + 1b
    X=X_without_noise+npr.rand(*X_without_noise.shape)*noise  # FG^T +a1 +1b + noise

    return model,X,X_without_noise

def simulation(R,C,J,spatial_dims,num_spots,blursize,noise,rho=None,anoise=.1,bnoise=.1,lam=0,lo=1e-10,varphinoise=.1):
    spatial_dims=tuple(spatial_dims)
    dims=spatial_dims+(R,C)
    dims_Y=spatial_dims+(J,)

    if rho is None:
        rho=np.exp(npr.randn(C))

    # sampel alpha
    alpha=np.ones((R,C)) + npr.rand(R,C)*.1

    # sample g
    g = np.eye(C) + npr.rand(C,C)*varphinoise

    # sample ground truth Fs
    F = np.zeros(dims_Y)
    F[tuple([npr.randint(0,x,size=num_spots) for x in dims_Y])]=20
    K=blurkernels.ContiguousBlur(spatial_dims,blursize)
    F_blurred =K@F

    # sample B
    B=np.zeros((R,C,J),dtype=np.bool)
    for j in range(J):
        for r in range(R):
            B[r,npr.randint(0,C),j]=True

    # make G
    Z=helpers.phasing(B,rho)
    Gtilde = np.einsum('ck, rkj -> rkj',g,Z)
    G = Gtilde*alpha[:,:,None]

    # make a b
    a=npr.rand(*spatial_dims)*anoise
    b=npr.rand(R,C)*bnoise

    # make noiseless data
    Frav = F_blurred.reshape((-1,J))
    Grav = G.reshape((-1,J))
    X_without_noise = (Frav@Grav.T + a.ravel()[:,None] + b.ravel()[None,:]).reshape(dims)
    X = X_without_noise+npr.randn(*dims)*noise

    F=F.reshape((-1,J))
    a=a.ravel()

    return parameters.Model(B,K,F=F,a=a,b=b,alpha=alpha,rho=rho,varphi=g,lo=lo,lam=lam),X,X_without_noise
