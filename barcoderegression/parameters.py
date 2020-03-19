import numpy as np
import scipy as sp
import scipy.ndimage

import scipy.optimize
import scipy.linalg
import numpy.linalg
import re
import numpy.random as npr

from .helpers import optional,nonnegative_update,clip,phasing,loss_at_q

class Model:
    def __init__(self,B,K,M=None,F=None,a=None,b=None,alpha=None,rho=None,varphi=None,
                        lo=1e-10,lam=0):
        '''
        A Model object holds the parameters for our model

        Input:
        - B -- binary barcode (R x C x J)
        - K -- picklable blur kernel
        - [optional] M -- number of voxels
        - [optional] F -- ndarray of shape M
        - [optional] a -- ndarray of shape M
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

        self.B=np.require(B)
        self.R,self.C,self.J=self.B.shape
        self.K=K
        self.lo=lo
        self.lam=lam

        if len(self.B.shape)!=3 or B.dtype!=np.bool:
            B_shape_error_message=re.sub("\s+"," ",fr'''
                B is expected to be a 3-dimensional boolean numpy array.
                B[r,c,j] is supposed to indicate whether gene "j" should appear
                bright in round "r" and channel "c".  Instead, we got an object
                with shape {B.shape} and type {B.dtype}
            ''')
            raise ValueError(B_shape_error_message)

        # handle M and F initialization
        if M is None:
            assert F is not None,'If M is not provided, F must be provided'
            self.F=np.require(F,dtype=np.float)
            self.M=F.shape[0]
            assert F.shape==(self.M,self.J),f'Shape of F must be {(self.M,self.J)} but was given {self.F.shape}'
        elif F is None:
            assert M is not None,'If F is not provided, M must be provided'
            self.M=M
            self.F=np.zeros((self.M,self.J))
        self.F_blurred=self.K@self.F

        # handle all the other initializations
        self.a=optional(a,np.zeros(self.M),'a')
        self.b=optional(b,np.zeros((self.R,self.C)),'b')
        self.alpha=optional(alpha,np.ones((self.R,self.C)),'alpha')
        self.varphi=optional(varphi,np.eye(self.C),'varphi')
        self.rho=optional(rho,np.zeros(self.C),'rho')

        self.nobs = self.M*self.R*self.C

    # code for saving parameters
    _props = ['F','a','b','alpha','varphi','rho','K','B']
    def snapshot(self):
        return {x:getattr(self,x) for x in self._props}

    def copy(self):
        snap=self.snapshot()
        snap={x:snap[x].copy() for x in snap}
        return Model(**snap)

    # intensity scaled to show total contribution of a gene to the original images
    def F_scaled(self,blurred=False):
        framel1=np.sum(np.sum(self.frame_loadings(),axis=0),axis=0)
        if blurred:
            return framel1[None,:] * self.F_blurred
        else:
            return framel1[None,:] * self.F

    # (perturbable) reconstructions
    def Z(self,rho=None):
        return phasing(self.B,optional(rho,self.rho))
    def frame_loadings(self,rho=None,alpha=None,varphi=None,F_blurred=None):
        alpha=optional(alpha,self.alpha)
        varphi=optional(varphi,self.varphi)
        return alpha[:,:,None]*np.einsum('ck, rkj -> rkj',varphi,self.Z(rho))
    def gene_reconstruction(self,rho=None,alpha=None,varphi=None):
        frame_loadings = self.frame_loadings(rho=rho,alpha=alpha,varphi=varphi)
        return np.einsum('mj,rcj->mrc',self.F_blurred,frame_loadings)
    def ab_reconstruction(self,a=None,b=None):
        return optional(b,self.b)[None,:,:] + optional(a,self.a)[:,None,None]
    def reconstruction(self,a=None,b=None,rho=None,alpha=None,varphi=None,F_blurred=None):
        return self.ab_reconstruction(a=a,b=b) + self.gene_reconstruction(rho=rho,alpha=alpha,
            varphi=varphi)

    # a helpful thing to be able to compute
    def FbmixedZ(self):
        '''
        FbmixedZ[m,r,c] = sum_jc' F_blurred[m,j] * varphi[c,c'] * Z[r,c',j]
        '''

        mixedZ = np.einsum('ck, rkj -> rkj',self.varphi,self.Z())
        FbmixedZ = np.einsum('rcj,mj -> mrc',mixedZ,self.F_blurred)
        return FbmixedZ

    # loss
    def loss(self,X,a=None,b=None,rho=None,alpha=None,varphi=None,F_blurred=None,lam=None):
        ab_recon = self.ab_reconstruction(a=a,b=b)
        gene_recon = self.gene_reconstruction(rho=rho,alpha=alpha,varphi=varphi)
        lam=optional(lam,self.lam)

        reconstruction_loss = .5*np.sum((X-ab_recon - gene_recon)**2)
        l1_loss = np.sum(gene_recon)


        lossinfo= dict(
            reconstruction = reconstruction_loss,
            l1 = l1_loss,
            lam=lam,
        )
        lossinfo['loss'] = (lossinfo['reconstruction'] + lam*lossinfo['l1'])/self.nobs

        return lossinfo

    # the updates!
    def update_a(self,X):
        resid = X - (self.gene_reconstruction() +self.b[None,:,:]) # M x R x C
        self.a = clip(np.mean(np.mean(resid,axis=1),axis=1))

    def update_b(self,X):
        resid = X - (self.gene_reconstruction() +self.a[:, None,None])
        self.b = clip(np.mean(resid,axis=0))

    def update_F(self,X):
        N=self.R*self.C
        F=self.F

        # just the
        frame_loadings = self.frame_loadings().reshape((N,self.J))

        # collect some things we'll need later
        framel1 = np.sum(frame_loadings,axis=0)
        framel2 = np.sum(frame_loadings**2,axis=0)
        riemannian = frame_loadings.T@ frame_loadings

        # updating Y, one column at at time (in a random order)
        for j in npr.permutation(self.J):
            '''
            get the quadratic form relevant to F[:,j]
            '''

            recon = self.reconstruction().reshape((self.M,N))
            recon_without_j_loadings = recon - np.outer(self.F_blurred[:,j],frame_loadings[:,j])
            residual_without_j_loadings = X.reshape((self.M,N)) - recon_without_j_loadings

            def apply_Gamma(x):
                return framel2[j]*(self.K.H@(self.K@x))
            phi = self.K @ (residual_without_j_loadings - self.lam) @ frame_loadings[:,j]

            # make the update
            F[:,j] = nonnegative_update(apply_Gamma,phi,F[:,j])
            self.F_blurred[:,j] = self.K@F[:,j]

    def update_rho(self,X,attempts=3):
        for c in npr.permutation(self.C):
            # consider some different values of q (modifying only q[c])
            options=np.r_[self.rho[c],np.unique(np.clip(.05*(npr.rand(attempts)-.5) + self.rho[c],0,1))]

            newrhos=[]
            for newval in options:
                newrho = self.rho.copy()
                newrho[c] = newval
                newrhos.append(newrho)

            # evaluate each one
            losses = [self.loss(X,rho=rho)['loss'] for rho in newrhos]

            # pick the best
            self.rho=newrhos[np.argmin(losses)]

    def update_alpha(self,X):
        # get the update
        Xmabl = X - self.ab_reconstruction() - self.lam
        FbmixedZ=self.FbmixedZ()
        numerator = np.einsum('mrc,mrc->rc',FbmixedZ,Xmabl)
        denom = np.sum(FbmixedZ**2,axis=0)

        # handle possibly zero denominators
        good = denom>self.lo
        alpha=self.alpha.copy()
        alpha[good] = numerator[good]/denom[good]

        # do cliping
        self.alpha=clip(alpha,self.lo)
