import numpy as np
import scipy as sp
import scipy.ndimage

import scipy.optimize
import scipy.linalg
import numpy.linalg
import re
import numpy.random as npr

from .helpers import optional,nonnegative_update,clip,phasing,quadratic_form_to_nnls_form

class Model:
    def __init__(self,B,K,M=None,F=None,a=None,b=None,alpha=None,rho=None,varphi=None,
                        lo=1e-10,lam=0,sigsq=None):
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
        - [optional] sigsq -- ndarray of shape R x C

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
        self.a=optional(a,lambda: np.zeros(self.M),'a')
        self.b=optional(b,lambda: np.zeros((self.R,self.C)),'b')
        self.alpha=optional(alpha,lambda: np.ones((self.R,self.C)),'alpha')
        self.varphi=optional(varphi,lambda: np.eye(self.C),'varphi')
        self.rho=optional(rho,lambda: np.zeros(self.C),'rho')
        self.sigsq=optional(sigsq,lambda: np.ones((self.R,self.C)),'sigsq')

        self.nobs = self.M*self.R*self.C

    # code for saving parameters
    _props = ['F','a','b','alpha','varphi','rho','B']
    _immutable_props = ['K','lam','lo']
    def snapshot(self):
        return {x:getattr(self,x) for x in (self._props+self._immutable_props)}

    def copy(self):
        snap = {x:getattr(self,x).copy() for x in self._props}
        snap.update({x:getattr(self,x) for x in self._immutable_props})
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
        return phasing(self.B,optional(rho,lambda: self.rho))
    def frame_loadings(self,rho=None,alpha=None,varphi=None,F_blurred=None):
        alpha=optional(alpha,lambda: self.alpha)
        varphi=optional(varphi,lambda: self.varphi)
        return np.einsum('rc,ck, rkj -> rcj',alpha,varphi,self.Z(rho))
    def gene_reconstruction(self,rho=None,alpha=None,varphi=None):
        frame_loadings = self.frame_loadings(rho=rho,alpha=alpha,varphi=varphi)
        return np.einsum('mj,rcj->mrc',self.F_blurred,frame_loadings)
    def ab_reconstruction(self,a=None,b=None):
        return optional(b,lambda: self.b)[None,:,:] + optional(a,lambda: self.a)[:,None,None]
    def reconstruction(self,a=None,b=None,rho=None,alpha=None,varphi=None,F_blurred=None):
        return self.ab_reconstruction(a=a,b=b) + self.gene_reconstruction(rho=rho,alpha=alpha,
            varphi=varphi)

    # a helpful thing to be able to compute
    def FbmixedZ(self):
        '''
        FbmixedZ[m,r,c] = sum_jc' F_blurred[m,j] * varphi[c,c'] * Z[r,c',j]
        '''

        mixedZ =np.einsum('ck, rkj -> rcj',self.varphi,self.Z(self.rho))
        FbmixedZ = np.einsum('rcj,mj -> mrc',mixedZ,self.F_blurred)
        return FbmixedZ

    # loss
    def loss(self,X,a=None,b=None,rho=None,alpha=None,varphi=None,F_blurred=None,lam=None):
        ab_recon = self.ab_reconstruction(a=a,b=b) # a1 + 1b
        gene_recon = self.gene_reconstruction(rho=rho,alpha=alpha,varphi=varphi) # KFG
        lam=optional(lam,lambda: self.lam)

        reconstruction_loss = .5*np.sum((X-ab_recon - gene_recon)**2/(self.sigsq[None,:,:])) # data
        reconstrution_loss = reconstruction_loss +.5*np.sum(np.log(2*np.pi*self.sigsq))*self.M # normalizer
        l1_loss = np.sum(gene_recon)  # L1_loss = |KFG^T|_1

        lossinfo= dict(
            reconstruction = reconstruction_loss,
            l1 = l1_loss,
            lam=lam,
        )
        lossinfo['undivided_loss']=lossinfo['reconstruction'] + lam*lossinfo['l1']
        lossinfo['loss'] = lossinfo['undivided_loss']/self.nobs

        return lossinfo

    # the updates!
    def update_a(self,X):
        resid = X - (self.gene_reconstruction() +self.b[None,:,:]) # M x R x C

        resid = resid / self.sigsq[None,:,:]
        resid = np.sum(np.sum(resid,axis=1),axis=1) # M
        resid = resid / np.sum(self.sigsq)

        self.a = clip(resid) # M

    def update_b(self,X):
        resid = X - (self.gene_reconstruction() +self.a[:, None,None]) # M x R x C
        self.b = clip(np.mean(resid,axis=0)) #  R x C

    def update_F(self,X):
        N=self.R*self.C
        F=self.F

        # just the
        G = self.frame_loadings().reshape((N,self.J))

        # collect fr things we'll need later
        ooss=1.0/self.sigsq.ravel() # N
        framel1 = np.sum(G,axis=0)
        framel2 = np.sum(ooss[:,None] * G**2,axis=0)

        # xmab= (X - self.ab_reconstruction()).reshape((self.M,N))
        # xmab_oss_ml = (xmab / oss[None,:]) - lam
        xmabl=(X - self.ab_reconstruction()).reshape((self.M,N)) - self.lam
        riemannian = G.T@ G


        # updating Y, one column at at time (in a random order)
        for j in npr.permutation(self.J):
            '''
            get the quadratic form relevant to F[:,j]

            loss(F[:,j]) = .5*F[:,j].T Gamma F[:,j] - phi^T F[:,j]

            basically this:
                recon = self.reconstruction().reshape((self.M,N))
                recon_without_j_loadings = recon - np.outer(self.F_blurred[:,j],G[:,j])
                residual_without_j_loadings = (X.reshape((self.M,N)) - recon_without_j_loadings) / ooss[None,:]
                phi = self.K @ (residual_without_j_loadings - self.lam) @ G[:,j]

                Gamma x = framel2[j]*(self.K.H@(self.K@x))

            but that's kind of slow to do for every j.  so, instead:

                phi = self.K @ (xmab/oss - recon_without_j_loadings - lam) @ G[:,j]
            '''

            phi = self.K@(xmabl @ G[:,j] - self.F_blurred@riemannian[:,j] + self.F_blurred[:,j]*framel2[j])

            if self.K.trivial:
                F[:,j] = np.clip(phi / framel2[j],0,None)
            else:
                def apply_Gamma(x):
                    return framel2[j]*(self.K.H@(self.K@x))
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

    def update_varphi(self,X):
        Z=self.Z() # R x C x J
        xmabl = X - self.ab_reconstruction() - self.lam # M x R x C
        F=self.F_blurred # M x J

        FZ = np.einsum('mj,rcj->mrc',F,Z)
        FZ_gamma = np.einsum('mrc,mrk->rck',FZ,FZ)

        for c1 in range(self.C):
            c1=npr.randint(0,self.C)
            Gamma_c = np.einsum('r,rck->ck',self.alpha[:,c1]**2,FZ_gamma)
            phi_c = np.einsum('r,mr,mrc->c',self.alpha[:,c1],xmabl[:,:,c1],FZ)
            A,b=quadratic_form_to_nnls_form(Gamma_c,phi_c)

            # TODO: enforce varphi[i,i]=1

            self.varphi[c1]= sp.optimize.nnls(A,b)[0]
