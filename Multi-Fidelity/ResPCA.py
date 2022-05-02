# residual PCA 
# 
# Author: Wei W. Xing (wxing.me)
# Email: wayne.xingle@gmail.com
# Date: 2022-05-02
# 

import torch

class PCA(object):
    def __init__(self, Y, r=0.99):
        # centerizing data
        self.mean = torch.mean(Y, axis=0)
        Y = Y - self.mean
        
        U, S, Vh = torch.linalg.svd(Y, full_matrices=True)
        cumuEnergy = S.cumsum(dim=0) / S.sum(dim=0)
        
        if r >= 1:
            rank = r 
        if r < 1:
            rank = (cumuEnergy>r).nonzero()[0][0]

        self.rank = rank
        # truncate the singular values and vector 
        U = U[:,0:rank]
        S = S[0:rank]
        Vh = Vh[0:rank,:]
        
        self.U = U
        self.S = S
        self.Vh = Vh
        
        self.Z = U @ S.diag_embed()

    def project(self, X):
        X = X - self.mean
        return X @ self.Vh.t()
        
    def recover(self, Z):
        Y = Z @ self.Vh
        return Y + self.mean.expand_as(Y)
    


class listPCA(object):
    # A PCA for a list of data
    def __init__(self, Ylist, r=0.99):
        # nData = len(Ylist)
        self.model_list = []
        for Y in Ylist:
        # for i in range(len(Ylist)):
            model = PCA(Y, r=r)
            self.model_list.append(model)
    
    def project(self, Xlist):
        Zlist = []       
        for i in range(len(Xlist)):
            Zlist.append(self.model_list[i].project(Xlist[i]))
            
        return Zlist
    
    def recover(self, Zlist):
        Ylist = []
        for i in range(len(Zlist)):
            Ylist.append(self.model_list[i].recover(Zlist[i]))
        return Ylist
    
class resPCA(object):
    # residual PCA
    def __init__(self, Ylist, r=0.99):
        for i in range(len(Ylist)):
            if i == 0:
                Ylist[i] = Ylist[i] #for i = 0
            else:
                Ylist[i] = Ylist[i] - Ylist[i-1]

        self.model = listPCA(Ylist)
        
    def project(self, Ylist):
        for i in range(len(Ylist)):
            if i == 0:
                Ylist[i] = Ylist[i] #for i = 0
            else:
                Ylist[i] = Ylist[i] - Ylist[i-1]
        Zlist = self.model.project(Ylist)
        return Zlist
        
    def recover(self, Zlist):
        Ylist = self.model.recover(Zlist)
        for i in range(len(Ylist)):
            if i == 0:
                Ylist[i] = Ylist[i] #for i = 0
            else:
                Ylist[i] = Ylist[i] + Ylist[i-1]
        return Ylist

    
# %% testing
from sklearn.metrics import mean_squared_error, r2_score
if __name__ == "__main__":
    print('testing')
    print(torch.__version__)
    
    Ytr = torch.randn(100, 10)
    Yte = torch.randn(200, 10)
    
    model = PCA(Ytr, r=3)
    Zte = model.project(Yte)
    Yte_pred = model.recover(Zte)
    print(r2_score(Yte,Yte_pred))

    model = PCA(Ytr, r=0.9)
    Zte = model.project(Yte)
    Yte_pred = model.recover(Zte)
    print(r2_score(Yte,Yte_pred))
    
    # listPCA
    print('testing listPCA')
    nlist = 3
    Ytr = []
    Yte = []
    
    for i in range(nlist):
        Ytr.append(torch.randn(100, 10))
        Yte.append(torch.randn(200, 10))
        
    model = listPCA(Ytr, r=0.99)
    Zte = model.project(Yte)
    Yte_pred = model.recover(Zte)
    
    r2 = []
    for i in range(nlist):
        print(r2_score(Yte[i], Yte_pred[i]))
        r2 = r2 + r2_score(Yte[i],Yte_pred[i])
    print(r2/nlist)
    
    # resPCA
    print('testing resPCA')
    nlist = 3
    Ytr = []
    Yte = []
    
    for i in range(nlist):
        Ytr.append(torch.randn(100, 10))
        Yte.append(torch.randn(200, 10))
        
    model = resPCA(Ytr, r=0.999)
    Zte = model.project(Yte)
    Yte_pred = model.recover(Zte)
    
    r2 = []
    for i in range(nlist):
        print(r2_score(Yte[i], Yte_pred[i]))
        r2 = r2 + r2_score(Yte[i],Yte_pred[i])
    print(r2/nlist)