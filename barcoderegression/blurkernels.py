import numpy as np
import scipy as sp
import scipy.ndimage

class ContiguousBlur:
    def __init__(self,dims,blurs):
        self.dims=np.array(dims)
        self.blurs=np.array(blurs)
        self.ndim=len(self.blurs)
        assert (self.blurs>=0).all(),'blur sizes should be nonnegative'
        assert len(self.dims.shape)==1,'dims should be a list of integers'
        assert self.dims.dtype==np.int
        assert len(self.dims)==len(self.blurs),'number of spatial dims should be same as length of blurs'

    @property
    def H(self):
        return self

    def copy(self):
        return self

    def __matmul__(self,x):
        x=np.require(x,dtype=np.float)
        xsh=x.shape
        x=x.reshape(tuple(self.dims)+(-1,))
        blurp = np.r_[self.blurs,0]
        x=sp.ndimage.gaussian_filter(x,blurp,mode='constant')
        return x.reshape(xsh)
