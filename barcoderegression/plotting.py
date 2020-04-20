import matplotlib.pylab as plt
import numpy as np

def plot_raw_data_2d(X):
    X=np.require(X)
    assert len(X.shape)==4,f"shape should be (spatialdim1,spatialdim2,rows,cols), but was given {X.shape}"

    R,C=X.shape[-2:]
    with AnimAcross(columns=C) as a:
        for r in range(R):
            for c in range(C):
                ~a
                if r==0:
                    plt.title(f"Ch:{c}",fontsize=30)
                if c==0:
                    plt.ylabel(f"Ch:{c}",fontsize=30)

                plt.imshow(X[:,:,r,c],vmin=0,vmax=X.max())

def plotspot(X,loc,barcode=None,radius=10,sz=1):
    loc=np.array(loc)
    st=np.max([np.zeros(len(loc)),loc-radius],axis=0)
    center = loc-st
    spdims=np.array(X.shape[:len(loc)])
    en=np.min([spdims,loc+radius+1],axis=0)
    sl=tuple([slice(int(a),int(b)) for (a,b) in zip(st,en)])

    R,C=X.shape[-2:]

    Xsub=X[sl]

    if (barcode is not None) and (np.sum(barcode,axis=1)==1).all():
        order=np.argsort(-1.0*barcode,axis=1)
    else:
        order=np.tile(np.r_[0:C],(R,1))

    R=X.shape[-2]
    C=X.shape[-1]

    with AnimAcross(columns=C,sz=sz) as a:
        for r in range(R):
            for c in range(C):
                c=order[r,c]
                ~a
                plt.imshow(Xsub[:,:,r,c])
                plt.axhline(center[0],color='red')
                plt.axvline(center[1],color='red')
                plt.xticks([]); plt.yticks([])

                if (barcode is not None) and barcode[r,c]:
                    for spine in plt.gca().spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(10)
                    # plt.plot([center[1]],[center[0]],'ro',ms=20,alpha=.5)


    plt.gcf().add_axes([C,1,3,3],label="centerline")
    center_vals= X[tuple(loc)].reshape((R,C))
    center_means=Xsub
    center_stds=Xsub
    for i in range(len(spdims)):
        center_means=np.mean(center_means,axis=0)
        center_stds=np.std(center_stds,axis=0)
    measurements=(center_vals - center_means) / center_stds
    measurements=np.array([m[x] for (x,m) in zip(order,measurements)])
    plt.imshow(measurements,vmin=0,vmax=3)
    plt.axis('off')

    plt.gcf().add_axes([C,4.2,3,3],label="centerline")
    center_vals= X[tuple(loc)].reshape((R,C))
    center_means=X
    center_stds=X
    for i in range(len(spdims)):
        center_means=np.mean(center_means,axis=0)
        center_stds=np.max(center_stds,axis=0)
    measurements=center_vals / center_stds
    measurements=np.array([m[x] for (x,m) in zip(order,measurements)])
    plt.imshow(measurements,vmin=0,vmax=1)
    plt.axis('off')

class AnimAcross:
    def __init__(self,ratio=.8,sz=4,columns=None,aa=None):
        self.aa=aa
        self.axes_list=[]
        self.cbs={}
        self.ratio=ratio
        self.sz=sz
        self.columns=columns

    def __enter__(self):
        if self.aa is not None:
            return self.aa
        else:
            return self

    def __invert__(self):
        self.axes_list.append(plt.gcf().add_axes([0,0,self.ratio,self.ratio],label="axis%d"%len(self.axes_list)))

    def __neg__(self):
        self.axes_list.append(plt.gcf().add_axes([0,0,self.ratio,self.ratio],label="axis%d"%len(self.axes_list)))
        plt.axis('off')

    def __call__(self,s,*args,**kwargs):
        ~self;
        plt.title(s,*args,**kwargs)

    def cb(self,mappable,idx=None):
        if idx is None:
            idx = len(self.axes_list)-1
        self.cbs[idx] = mappable

    def __exit__(self,exc_type,exc_val,exc_tb):
        if self.aa is not None:
            return

        if self.columns is None:
            dims=[
                (1,1), # no plots
                (1,1), # 1 plot
                (1,2), # 2 plots
                (1,3), # 3 plots
                (2,2), # 4 plots
                (2,3), # 5 plots
                (2,3), # 6 plots
                (3,3), # 7 plots
                (3,3), # 8 plots
                (3,3), # 9 plots
                (4,4)
            ]
            if len(self.axes_list)<len(dims):
                dims=dims[len(self.axes_list)]
            else:
                cols=int(np.sqrt(len(self.axes_list)))+1
                rows = len(self.axes_list)//cols + 1
                dims=(rows,cols)
        else:
            cols=self.columns
            if len(self.axes_list)%cols==0:
                rows=len(self.axes_list)//cols
            else:
                rows=len(self.axes_list)//cols + 1
            dims=(rows,cols)

        plt.gcf().set_size_inches(self.sz,self.sz)
        k=0

        for j in range(dims[0]):
            for i in range(dims[1]):
                if k<len(self.axes_list):
                    self.axes_list[k].set_position((i,dims[0]-j,self.ratio,self.ratio))
                k=k+1

        for i in range(len(self.axes_list)):
            if i in self.cbs:
                plt.colorbar(mappable=self.cbs[i],ax=self.axes_list[i])

        if exc_type is not None:
            print(exc_type,exc_val,exc_tb)
