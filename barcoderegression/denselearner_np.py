import numpy as np
import scipy as sp
import scipy.ndimage
from . import blurkernels

import scipy.optimize
import scipy.linalg
import numpy.linalg
import re
import numpy.random as npr

from . import helpers


def heat_kernel(X,niter,axis=0):
    X=np.require(X)
    X=np.swapaxes(X,axis,0)
    pad_width=[(0,0) for i in range(len(X.shape))]
    pad_width[0]=(niter,niter)
    X=np.pad(X,pad_width)
    for i in range(niter*2):
        X=.5*(X[1:]+X[:-1])
    X=np.swapaxes(X,axis,0)
    return X
def heat_kernel_nd(X,niters,axis=0):
    for i in range(len(niters)):
        X=heat_kernel(X,niters[i],axis=i)
    return X

class HeatKernel:
    def __init__(self,spatial_dims,blur_level):
        self.spatial_dims=spatial_dims
        self.nspatial=len(self.spatial_dims)

        if blur_level is None:
            self.blur_level=None
        elif isinstance(blur_level,int):
            self.blur_level=np.ones(self.nspatial,dtype=np.int)*blur_level
        else:
            self.blur_level=np.array(blur_level)
            assert self.blur_level.dtype==np.int64
            assert self.blur_level.shape==(self.nspatial,)

    def __matmul__(self,X):
        if self.blur_level is None:
            return X
        else:
            return heat_kernel_nd(X,list(self.blur_level))

class GaussianBlurKernel:
    def __init__(self,spatial_dims,blur_level):
        self.spatial_dims=spatial_dims
        self.nspatial=len(self.spatial_dims)
        self.blur_level=blur_level
        if blur_level is not None:
            self.blur_level=np.require(self.blur_level,dtype=np.float32)
            assert self.blur_level.shape==(self.nspatial,) or self.blur_level.shape==()

    def __matmul__(self,X):
        if self.blur_level is None:
            return X
        else:
            x=np.require(X,dtype=np.float)
            xsh=x.shape
            x=x.reshape(tuple(self.spatial_dims)+(-1,))
            blurp = np.r_[self.blur_level,0]
            x=sp.ndimage.gaussian_filter(x,blurp,mode='constant')
            return x.reshape(xsh)

class Model:
    def __init__(self,codebook,spatial_dims,blur_level=None,F=None,a=None,b=None,alpha=None,rho=None,varphi=None,
                        lo=1e-10,lam=0):
        '''
        A Model object holds the parameters for our model

        Input:
        - codebook -- binary codebook (R x C x J)
        - spatial_dims -- i.e. (npix_X,npix_Y) for 2d data or (npix_X,npix_Y,npix_Z) for 3d data
        - [optional] blur_level -- how much gaussian blur
        - [optional] F -- ndarray of shape spatial_dims
        - [optional] a -- ndarray of shape spatial_dims
        - [optional] b -- ndarray of shape R x C
        - [optional] alpha -- ndarray of shape R x C
        - [optional] rho -- ndarray of shape C
        - [optional] varphi -- ndarray of shape C x C
        - [optional] lo -- scalar, smallest possible value of alpha
        - [optional] lam -- magnitude of L1 penalty on gene reconstruction

        If the optional parameters are not given, they will be initialized
        automatically.  One caveat to this is F and M -- one or the other
        must be provided:
        - If F is given then M is not required.
        - If M is given then F is not required.
        - If both are given their shapes should agree.
        - If neither are given an exception will be thrown.
        '''

        self.codebook=np.require(codebook)
        self.spatial_dims=tuple(spatial_dims)
        assert len(self.spatial_dims) in [1,2,3]
        self.nspatial=len(self.spatial_dims)
        self.R,self.C,self.J=self.codebook.shape

        self.K=HeatKernel(spatial_dims,blur_level)
        # self.K=GaussianBlurKernel(spatial_dims,blur_level)

        self.lo=lo
        self.lam=lam

        if len(self.codebook.shape)!=3:
            B_shape_error_message=fr'''
                B is expected to be a 3-dimensional boolean numpy array.
                B[r,c,j] is supposed to indicate whether gene "j" should appear
                bright in round "r" and channel "c".  Instead, we got an object
                with shape {B.shape} and type {B.dtype}
            '''
            raise ValueError(helpers.kill_whitespace(B_shape_error_message))

        # handle all the other initializations
        self.F=helpers.optional(F,self.spatial_dims+(self.J,),np.zeros)
        self.a=helpers.optional(a,(self.spatial_dims),np.zeros)
        self.b=helpers.optional(b,(self.R,self.C),np.zeros)
        self.alpha=helpers.optional(alpha,(self.R,self.C),np.ones)
        self.varphi=helpers.optional_eye(varphi,self.C)
        self.rho=helpers.optional(rho,(self.C,),np.zeros)

        # calc some things we'll need later
        self.M=np.prod(self.spatial_dims)
        self.N=self.R*self.C
        self.F_blurred=self.K@self.F
        self.nobs = self.M*self.N

    # code for saving parameters
    _props = ['codebook','spatial_dims','K','F','a','b','alpha','rho','varphi','lo','lam']
    def snapshot(self):
        return {x:getattr(self,x) for x in self._props}
    def copy(self):
        snap = {x:getattr(self,x).copy() for x in self._props}
        return Model(**snap)

    # intensity scaled to show total contribution of a gene to the original images
    def F_scaled(self,blurred=False):
        framel1=np.sum(np.sum(self.frame_loadings(),axis=0),axis=0)
        if blurred:
            return framel1[None,:] * self.F_blurred
        else:
            return framel1[None,:] * self.F

    # reconstructions
    def Z(self):
        return helpers.phasing(self.codebook,self.rho)
    def frame_loadings(self):
        return np.einsum('rc,ck, rkj -> rcj',self.alpha,self.varphi,self.Z())
    def gene_reconstruction(self,rho=None,alpha=None,varphi=None):
        frame_loadings = self.frame_loadings()
        return np.einsum('...j,rcj->...rc',self.F_blurred,frame_loadings)
    def a_broadcast(self):
        sl = (len(self.spatial_dims)*(slice(0,None),)) + ((None,)*2)
        return self.a[sl]
    def b_broadcast(self):
        sl = (len(self.spatial_dims)*(None,)) + ((slice(0,None),)*2)
        return self.b[sl]
    def ab_reconstruction(self):
        return self.a_broadcast() + self.b_broadcast()
    def reconstruction(self):
        return self.ab_reconstruction()+self.gene_reconstruction()
    def FbmixedZ(self):
        '''
        FbmixedZ[m,r,c] = sum_jc' F_blurred[m,j] * varphi[c,c'] * Z[r,c',j]
        '''
        mixedZ =np.einsum('ck, rkj -> rcj',self.varphi,self.Z())
        FbmixedZ = np.einsum('rcj,...j -> ...rc',mixedZ,self.F_blurred)
        return FbmixedZ

    # loss
    def loss(self,X):
        ab_recon = self.ab_reconstruction() # a1 + 1b
        gene_recon = self.gene_reconstruction() # KFG

        reconstruction_loss = .5*np.sum((X-ab_recon - gene_recon)**2)
        l1_loss = np.sum(gene_recon)  # L1_loss = |KFG^T|_1

        lossinfo= dict(
            reconstruction = reconstruction_loss,
            l1 = l1_loss,
            lam=self.lam,
        )
        lossinfo['total_loss']=lossinfo['reconstruction'] + self.lam*lossinfo['l1']
        lossinfo['loss'] = lossinfo['total_loss']/self.nobs

        return lossinfo

    # the updates!
    def update_a(self,X):
        resid = X - (self.gene_reconstruction() + self.b_broadcast()) # spatial dims x R x C
        resid = np.mean(np.mean(resid,axis=-1),axis=-1) # spatial_dims
        self.a = np.clip(resid,0,None) # spatial_dims

    def update_b(self,X):
        resid = X - (self.gene_reconstruction() +self.a_broadcast()) # spatial_dims x R x C
        for i in range(len(self.spatial_dims)):
            resid=np.mean(resid,axis=0)
        self.b=resid  # R x C

    def update_F(self,X):
        F=self.F.reshape((self.M,self.J))
        G = self.frame_loadings().reshape((self.N,self.J))
        framel1 = np.sum(G,axis=0)
        framel2 = np.sum(G**2,axis=0)
        xmabl=(X - self.ab_reconstruction()).reshape((self.M,self.N)) - self.lam
        riemannian = G.T@ G

        '''
        loss = .5* ||X - ab - KFG^T ||^2 + lam*||KFG^T||_1
             = .5* ||KFG^T||^2 - tr((KFG^T) (X - ab - lam)^T)
             = .5* tr(KFG^T G F^T K) - tr(F G (X - ab - lam)^T K)
        '''

        def apply_K(x):
            return (self.K@x.reshape(self.spatial_dims + (-1,))).reshape((x.shape))
        def apply_Ksq(x):
            return apply_K(apply_K(x))
        def apply_Gamma(F):
            return apply_Ksq(F @ riemannian)

        linear_term = apply_K(xmabl @ G)

        # fook=self.K@np.ones((self.M,self.N))@G
        # oldloss = np.sum(F*fook)   # tr( F G^T 1 1^T K) = tr( K F G^T 1 1^T)
        # oldloss2 = np.sum((self.K@F)@G.T)
        # print("the L1s disagree",oldloss,oldloss2)

        F=helpers.nonnegative_update(apply_Gamma,linear_term,F)
        self.F = F.reshape(self.F.shape)
        self.F_blurred = self.K@self.F

        # newloss=np.sum(F*fook)
        # newloss2=self.loss(X)['l1']
        #
        # print('predicted change!',oldloss-newloss,oldloss2-newloss2)

    def update_alpha(self,X):
        # get the update
        Xmabl = (X - self.ab_reconstruction() - self.lam).reshape((self.M,self.R,self.C))
        FbmixedZ=self.FbmixedZ().reshape((self.M,self.R,self.C))
        numerator = np.einsum('mrc,mrc->rc',FbmixedZ,Xmabl)
        denom = np.sum(FbmixedZ**2,axis=0)

        # handle possibly zero denominators
        good = denom>self.lo
        alpha=self.alpha.copy()
        alpha[good] = numerator[good]/denom[good]

        # do cliping
        self.alpha=np.clip(alpha,self.lo,None)

    def update_varphi(self,X):
        Z=self.Z() # R x C x J
        xmabl = X - self.ab_reconstruction() - self.lam # spatial x R x C
        F=self.F_blurred # spatial x J

        xmabl=xmabl.reshape((self.M,self.R,self.C))
        F=F.reshape((self.M,self.J))

        FZ = np.einsum('mj,rcj->mrc',F,Z)
        FZ_gamma = np.einsum('mrc,mrk->rck',FZ,FZ)

        for c1 in range(self.C):
            Gamma_c = np.einsum('r,rck->ck',self.alpha[:,c1]**2,FZ_gamma)
            phi_c = np.einsum('r,mr,mrc->c',self.alpha[:,c1],xmabl[:,:,c1],FZ)
            A,b=helpers.quadratic_form_to_nnls_form(Gamma_c,phi_c)
            self.varphi[c1]= sp.optimize.nnls(A,b)[0]
