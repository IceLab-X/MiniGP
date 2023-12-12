
# Conditional independent Gaussian process (CIGP) for vector output regression based on pytorch
# 
# CIGP use a single kernel for each output. Thus the log likelihood is simply a sum of the log likelihood of each output.

# Author: Wei W. Xing (wxing.me)
# Email: wayne.xingle@gmail.com
# Date: 2023-11-26

import numpy as np
import torch
import torch.nn as nn
import kernel as kernel
import time as time

import gp_computation_pack as gp_pack
    
class CIGP(nn.Module):
    def __init__(self, kernel, noise_variance):
        super().__init__()
        self.kernel = kernel
        self.noise_variance = nn.Parameter(torch.tensor([noise_variance]))
        # define a mean function according to the input shape
        self.mean_func = nn.Sequential(
            nn.Linear(1, 3),
            # nn.ReLU(),
            # nn.Linear(10, 3)
        )

    def forward(self, x_train, y_train, x_test):
        K = self.kernel(x_train, x_train) + self.noise_variance.pow(2) * torch.eye(len(x_train))
        K_s = self.kernel(x_train, x_test)
        K_ss = self.kernel(x_test, x_test)
        mean_part_train = self.mean_func(x_train)
        mean_part_test = self.mean_func(x_test)
        
        mu, cov = gp_pack.conditional_Gaussian(y_train-mean_part_train, K, K_s, K_ss)
        return mu.squeeze()+ mean_part_test, cov

    def log_likelihood(self, x_train, y_train):
        K = self.kernel(x_train, x_train) + self.noise_variance.pow(2) * torch.eye(len(x_train))
        mean_part = self.mean_func(x_train)
        return gp_pack.Gaussian_log_likelihood(y_train-mean_part, K)
        
# downstate here how to use the GP model
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    print('testing')
    print(torch.__version__)

    # SIMO test 1
    torch.manual_seed(1)       #set seed for reproducibility
    xte = torch.linspace(-1, 7, 100).view(-1, 1)
    yte = torch.hstack([torch.sin(xte),
                       torch.cos(xte),
                        xte.tanh()] )

    xtr = torch.rand(32, 1) * 6
    ytr = torch.sin(xtr) + torch.rand(32, 1) * 0.5
    ytr = torch.hstack([torch.sin(xtr),
                       torch.cos(xtr),
                        xtr.tanh()] )+ torch.randn(32, 3) * 0.2
    
    # main
    # normalize output data to zero mean and unit variance across each dimension
    # ytr = (ytr - ytr.mean(0)) / ytr.std(0) # for cigp, this not work well
    # ytr = (ytr - ytr.mean()) / ytr.std()    #works well for cigp
    
    # define kernel function
    kernel1 = kernel.ARDKernel(1)
    # kernel1 = kernel.MaternKernel(1)   
    kernel1 = kernel.LinearKernel(1,-1.0,1.)   
    kernel1 = kernel.SumKernel(kernel.LinearKernel(1), kernel.MaternKernel(1))
    
    GPmodel = CIGP(kernel=kernel1, noise_variance=1.0)
    optimizer = torch.optim.Adam(GPmodel.parameters(), lr=1e-1)
    
    for i in range(1000):
        startTime = time.time()
        optimizer.zero_grad()
        loss = -GPmodel.log_likelihood(xtr, ytr)
        loss.backward()
        optimizer.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()))
        timeElapsed = time.time() - startTime
        print('time elapsed: {:.3f}s'.format(timeElapsed))
        
    with torch.no_grad():
        ypred, ypred_var = GPmodel.forward(xtr, ytr, xte)
        

    # plt.close('all')
    color_list = ['r', 'g', 'b']
    
    plt.figure()
    # plt.plot(xtr, ytr, 'b+')
    for i in range(3):
        plt.plot(xtr, ytr[:, i], color_list[i]+'+')
        # plt.plot(xte, yte[:, i], label='truth', color=color_list[i])
        plt.plot(xte, ypred[:, i], label='prediction', color=color_list[i], linestyle='--')
        plt.fill_between(xte.squeeze(-1).detach().numpy(),
                         ypred[:, i].squeeze(-1).detach().numpy() + torch.sqrt(ypred_var[:, i].squeeze(-1)).detach().numpy(),
                         ypred[:, i].squeeze(-1).detach().numpy() - torch.sqrt(ypred_var[:, i].squeeze(-1)).detach().numpy(),
                         alpha=0.2)
    plt.show()

