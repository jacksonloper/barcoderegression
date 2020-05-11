import numpy as np

class Trainer:
    def __init__(self,X,model):
        X=np.require(X,dtype=np.float)
        self.X=X
        self.model=model

        self.losses=[model.loss(self.X)]
        self.losses[-1]['improvement']=0
        self.losses[-1]['action']='initialization'

    def record_loss(self,nm):
        self.losses.append(self.model.loss(self.X))
        self.losses[-1]['action']=nm
        self.losses[-1]['improvement']=self.losses[-2]['loss'] - self.losses[-1]['loss']


    def update(self,nms,record_loss_every_change=False):
        for nm in nms:
            getattr(self.model,'update_'+nm)(self.X)
            if record_loss_every_change:
                self.record_loss(nm)
        if not record_loss_every_change:
            self.record_loss('endsweep: ' + '_'.join(nms))

    def train_tqdm_notebook(self,nms,iters,record_loss_every_change=False):
        import tqdm.notebook
        trange=tqdm.notebook.tqdm(range(iters))
        for i in trange:
            self.update(nms=nms,record_loss_every_change=record_loss_every_change)
            trange.set_description(f"{self.losses[-1]['loss']:.2e}")

    def status(self,print_bads=True):
        import matplotlib.pylab as plt
        overall_losses=[x['loss'] for x in self.losses]
        worst=np.diff(overall_losses).max()
        if worst<=0:
            print("we never went the wrong way!")
        else:
            print("we went wrong way",worst)
        plt.plot(overall_losses,'-o')
        plt.gca().set_yscale('log')

        lossinfo=self.model.loss(self.X)
        print('final reconstruction loss per observation',lossinfo['reconstruction']/self.model.nobs)
        print('final L1 loss per observation            ',self.model.lam*lossinfo['l1']/self.model.nobs)
        print('final loss per observation               ',lossinfo['loss'])

        if print_bads:
            bads=[x for x in self.losses if x['improvement']<0]
            for b in bads:
                print(b['action'],b['reconstruction'],b['improvement'])
