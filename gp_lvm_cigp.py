# gaussian process latent variable model (GPLVM) for dimensionality reduction based using cigp (see gp_cigp.py) as the GP model.
#
# Author: Wei W. Xing (wxing.me)
# Email: wayne.xingle@gmail.com
# Date: 2023-11-26


import numpy as np
import torch
import torch.nn as nn
import kernel as kernel
import time as time

class CIGP(nn.Module):
    def __init__(self, kernel, noise_variance):
        super().__init__()
        self.kernel = kernel
        self.noise_variance = nn.Parameter(torch.tensor([noise_variance]))

    def forward(self, x_train, y_train, x_test):
        K = self.kernel(x_train, x_train) + self.noise_variance.pow(2) * torch.eye(len(x_train))
        K_s = self.kernel(x_train, x_test)
        K_ss = self.kernel(x_test, x_test)
        
        # recommended implementation, fastest so far
        L = torch.cholesky(K)
        Alpha = torch.cholesky_solve(y_train, L)
        mu = K_s.T @ Alpha
        # v = torch.cholesky_solve(K_s, L)    # wrong implementation
        v = L.inverse() @ K_s   # correct implementation
        cov = K_ss - v.T @ v

        cov = cov.diag().view(-1, 1).expand_as(mu)
        return mu.squeeze(), cov
            
    def log_likelihood(self, x_train, y_train):
        K = self.kernel(x_train, x_train) + self.noise_variance.pow(2) * torch.eye(len(x_train))
        
        L = torch.linalg.cholesky(K)
        log_det_K = 2 * torch.sum(torch.log(torch.diag(L)))
        Alpha = torch.cholesky_solve(y_train, L, upper = False)
        
        # return - 0.5 * (Alpha.T @ Alpha + log_det_K + len(x_train) * np.log(2 * np.pi))
        return - 0.5 * ( (Alpha ** 2).sum() + log_det_K + len(x_train) * np.log(2 * np.pi))
        
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
    ytr = torch.sin(xtr) + torch.rand(64, 1) * 0.1
    ytr = torch.hstack([torch.sin(xtr),
                       torch.cos(xtr),
                        xtr.tanh()] )+ torch.randn(64, 3) * 0.1
    
    kernel1 = kernel.ARDKernel(1)
    # kernel1 = kernel.LinearKernel(1)
    kernel1 = kernel.SumKernel(kernel.LinearKernel(1), kernel.ARDKernel(1))
    
    # define the gplvm model
    # define latent variable
    latent_dim = 1
    z = nn.Parameter(torch.rand(len(xtr), latent_dim))
    
    # define GP model as if we are doing regression
    GPmodel = CIGP(kernel=kernel1, noise_variance=1.0)
    
    # optimization now includes the latent variable z
    optimizer = torch.optim.Adam( list(GPmodel.parameters()) + [z], lr=1e-1)
    # optimizer = torch.optim.Adam(GPmodel.parameters(), lr=1e-1)
    
    for i in range(1000):
        startTime = time.time()
        optimizer.zero_grad()
        loss = -GPmodel.log_likelihood(z, ytr)
        loss.backward()
        optimizer.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()))
        timeElapsed = time.time() - startTime
        print('time elapsed: {:.3f}s'.format(timeElapsed))
        
    # with torch.no_grad():
    #     ypred, ypred_var = GPmodel.forward(xtr, ytr, xte)
        
    plt.figure()
    plt.scatter(z.detach().numpy(), xtr.detach().numpy(), c='r', marker='+')
    plt.show()
        
    # # plt.close('all')
    # color_list = ['r', 'g', 'b']
    
    # plt.figure()
    # # plt.plot(xtr, ytr, 'b+')
    # for i in range(3):
    #     plt.plot(xtr, ytr[:, i], color_list[i]+'+')
    #     # plt.plot(xte, yte[:, i], label='truth', color=color_list[i])
    #     plt.plot(xte, ypred[:, i], label='prediction', color=color_list[i], linestyle='--')
    #     plt.fill_between(xte.squeeze(-1).detach().numpy(),
    #                      ypred[:, i].squeeze(-1).detach().numpy() + torch.sqrt(ypred_var[:, i].squeeze(-1)).detach().numpy(),
    #                      ypred[:, i].squeeze(-1).detach().numpy() - torch.sqrt(ypred_var[:, i].squeeze(-1)).detach().numpy(),
    #                      alpha=0.2)
    # plt.show()

