import numpy as np
import scipy as sp
import scipy.spatial


def calc_l2s(X,codebook,normalize='bystd'):
    '''
    Input:
    - X: spatial_dims x R x C
    - codebook: R x C x numgenes

    Output:
    - dsts: spatial_dims x numgenes

    dists[m,j] = the cosine distance between the data in voxel m and the barcode j
    '''
    codebook=np.require(codebook,dtype=np.float64)
    X=np.require(X,dtype=np.float64)
    spatial_dims=X.shape[:-2]

    R,C,J=codebook.shape
    N=R*C
    M=np.prod(spatial_dims)

    codebook=codebook.reshape((N,J)).T    # J x N
    X=X.reshape((M,N))  # M x N

    if normalize=='bystd':
        X=X/np.std(X,axis=0)
    elif normalize is None:
        pass
    else:
        raise Exception(f"dont know what {normalize} is;  acceptable options are None and 'bystd'")

    X=X/np.sqrt(np.sum(X**2,axis=1,keepdims=True))
    codebook=codebook/np.sqrt(np.sum(codebook**2,axis=1,keepdims=True))

    return sp.spatial.distance.cdist(X,codebook).reshape(spatial_dims+(J,))
