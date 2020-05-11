def update_stripout(self,X2):
    F=self.get_stripped_out_F(1)
    self.F=tf.convert_to_tensor(F,dtype=tf.float64)
    self.F_blurred=self.K@self.F
    for i in range(3):
        self.update_alpha(X2)
        self.update_varphi(X2)
        self.update_a(X2)
        self.update_b(X2)

def get_top_genes(self):
    fsc=self.F_scaled(blurred=False).numpy()
    best=np.argsort(fsc,axis=-1)
    return best

def get_stripped_out_F(model,n):
    F=model.F.numpy().copy()
    F_scaled=model.F_scaled(blurred=True).numpy().copy()
    mx=np.sort(F_scaled,axis=-1)[:,:,[-n]] # get best gene according to F_scaled
    good=F_scaled>=mx # we're happy when its the best gene
    F[~good]=0 # zero out the Fs that aren't the BEST!
    return F
