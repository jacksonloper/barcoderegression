import numpy as np
from . import parameters
from . import helpers

class Trainer:
    def __init__(self,X,model):
        X=np.require(X,dtype=np.float)
        self.X=X
        self.model=model
        self.Xrav=X.reshape((-1,self.model.R,self.model.C))

        self.losses=[model.loss(self.Xrav)]
        self.losses[-1]['improvement']=0
        self.losses[-1]['action']='initialization'

    def record_loss(self,nm):
        self.losses.append(self.model.loss(self.Xrav))
        self.losses[-1]['action']=nm
        self.losses[-1]['improvement']=self.losses[-2]['loss'] - self.losses[-1]['loss']
        self.check_for_nans()

    def check_for_nans(self):
        for nm in ['F','a','b','alpha','varphi','rho']:
            assert not np.isnan(getattr(self.model,nm)).any()

    def update(self,nms=['F','a','b','alpha']):
        for nm in nms:
            getattr(self.model,'update_'+nm)(self.Xrav)
            self.record_loss(nm)
