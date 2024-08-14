# gaussian process latent variable model (GPLVM) for dimensionality reduction based using cigp (see gp_cigp.py) as the GP model.
#
# Author: Wei W. Xing (wxing.me)
# Email: wayne.xingle@gmail.com
# Date: 2023-11-26
# 
# The idea is simple. Basically, we use a GP model to do regression, but the input is the latent variable z (to be learned) The output is the original data x. The rest of the code is almost identical to the GP regression model.


import numpy as np
import torch
import torch.nn as nn
from core.kernel import ARDKernel
import time as time
import core.GP_CommonCalculation as GP
EPS = 1e-10
JITTER= 1e-6
torch.set_default_dtype(torch.float64)
class gplvm(nn.Module):
    def __init__(self,X,Y,kernel=ARDKernel,latent_dim=1):
        super(gplvm,self).__init__()

        self.kernel = kernel(latent_dim)

        self.Z = nn.Parameter(torch.rand(len(X), latent_dim))
        self.normalizer=GP.DataNormalization(mode=1)
        self.normalizer.fit(Y,'y')
        self.X = X
        self.Y = self.normalizer.normalize(Y,'y')


        self.kernel = kernel(input_dim=X.size(1))

        # GP hyperparameters
        self.log_beta = nn.Parameter(torch.ones(1) * 1)
    def forward(self, Xte):

        K = self.kernel(self.X, self.X) + (self.log_beta.pow(2)) * torch.eye(len(self.X))
        K_s = self.kernel(self.X, Xte)
        K_ss = self.kernel(Xte, Xte)
        
        # recommended implementation, fastest so far
        L = torch.linalg.cholesky(K)
        Alpha = torch.cholesky_solve(self.Y, L)
        mu = K_s.T @ Alpha
        # v = torch.cholesky_solve(K_s, L)    # wrong implementation
        v = L.inverse() @ K_s   # correct implementation
        cov = K_ss - v.T @ v

        cov = cov.diag().view(-1, 1)
        #denormalize
        mu = self.normalizer.denormalize(mu,'y')
        #cov = self.normalizer.denormalize_cov(cov,'y')
        return mu.squeeze(), 1
            
    def negative_log_likelihood(self):
        K = self.kernel(self.Z, self.Z) + (self.log_beta.pow(2)) * torch.eye(len(self.Z))
        
        L = torch.linalg.cholesky(K)
        log_det_K = 2 * torch.sum(torch.log(torch.diag(L)))
        Alpha = torch.cholesky_solve(self.Y, L, upper = False)
        
        return  -0.5 * ((Alpha ** 2).sum() + 0.5*log_det_K + 0.5*len(self.X) * np.log(2 * np.pi))
    def train_adam(self, num_iter, lr=1e-1):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for i in range(num_iter):
            optimizer.zero_grad()
            loss = self.negative_log_likelihood()
            loss.backward()
            optimizer.step()
            if i+1 % 10 == 0:
                print('iter', i+1, 'nll:{:.5f}'.format(loss.item()))
            print('iter', i, 'nll:{:.5f}'.format(loss.item()))
        return loss.item()
        
# downstate here how to use the GP model
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    print('testing')
    print(torch.__version__)

    # SIMO test 1
    torch.manual_seed(1)       #set seed for reproducibility
    xte = torch.linspace(0, 6, 100).view(-1, 1)
    yte = torch.hstack([torch.sin(xte),
                       torch.cos(xte),
                        xte.tanh()] )

    xtr = torch.rand(64, 1) * 6
    #ytr = torch.sin(xtr) + torch.rand(64, 1) * 0.1
    ytr = torch.hstack([torch.sin(xtr),
                       torch.cos(xtr),
                        xtr.tanh()] )+ torch.randn(64, 3) * 0.1


    # kernel1 = kernel.LinearKernel(1)
    #kernel1 = kernel.SumKernel(kernel.LinearKernel(1), kernel.ARDKernel(1))
    
    # define the gplvm model
    # define latent variable
    latent_dim = 1

    
    # define GP model as if we are doing regression
    GPmodel = gplvm(xtr,ytr)
    
    # optimization now includes the latent variable z
    optimizer = torch.optim.Adam(GPmodel.parameters(), lr=1e-1)
    # optimizer = torch.optim.Adam(GPmodel.parameters(), lr=1e-1)

    for i in range(200):
        startTime = time.time()
        optimizer.zero_grad()
        loss = -GPmodel.negative_log_likelihood()
        loss.backward()
        optimizer.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()))
        timeElapsed = time.time() - startTime
        print('time elapsed: {:.3f}s'.format(timeElapsed))
        
    with torch.no_grad():
        ypred, ypred_var = GPmodel.forward(xte)

    plt.figure()
    plt.scatter(GPmodel.Z.detach().numpy(), xtr.detach().numpy(), c='r', marker='+')
    plt.show()
        
    # plt.close('all')
    color_list = ['r', 'g', 'b']

    plt.figure()
    # plt.plot(xtr, ytr, 'b+')
    for i in range(3):
        #plt.plot(xtr, ytr[:, i], color_list[i]+'+')
        #plt.scatter(GPmodel.Z.detach().numpy(), ytr[:, i], color=color_list[i])
        plt.plot(xte, yte[:, i], label='truth', color=color_list[i])
        plt.plot(xte, ypred[:, i], label='prediction', color=color_list[i], linestyle='--')
        # plt.fill_between(xte.squeeze(-1).detach().numpy(),
        #                  ypred[:, i].squeeze(-1).detach().numpy() + torch.sqrt(ypred_var[:, i].squeeze(-1)).detach().numpy(),
        #                  ypred[:, i].squeeze(-1).detach().numpy() - torch.sqrt(ypred_var[:, i].squeeze(-1)).detach().numpy(),
        #                  alpha=0.2)
    plt.legend()
    plt.show()

