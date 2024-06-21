# Conditional independent Gaussian process (CIGP) for vector output regression based on pytorch
# CIGP use a single kernel for each output. Thus the log likelihood is simply a sum of the log likelihood of each output.
#
# v10: A stable version. improve over the v02 version to fix nll bug; adapt to torch 1.11.0.
# v12: add mean function to v10
# v122: add mean function using any function as input to cigp
#
# Author: Wei W. Xing (wxing.me)
# Email: wayne.xingle@gmail.com
# Date: 2022-12-29


# %%
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

print(torch.__version__)
# I use torch (1.11.0) for this work. lower version may not work.

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Fixing strange error if run in MacOS
JITTER = 1e-6
EPS = 1e-10
PI = 3.1415

# linear regression function
class linearRegression(nn.Module):
    def __init__(self, X_dim, Y_dim):
        super(linearRegression, self).__init__()
        self.W = nn.Parameter(torch.ones(X_dim, Y_dim))
        self.b = nn.Parameter(torch.zeros(Y_dim))
        
    def forward(self, X):
        return X.mm(self.W) + self.b

# define a constant output function
class constFunc(nn.Module):
    def __init__(self, Y_dim):
        super(constFunc, self).__init__()
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        return self.b.expand(X.size(0), 1)
    
# always output zero function
class zeroFunc(nn.Module):
    def __init__(self, Y_dim):
        super(zeroFunc, self).__init__()

    def forward(self, X):
        return torch.zeros(X.size(0), 1)
        

class cigp(nn.Module):
    def __init__(self, X, Y, meanFunc=0, normal_y_mode=0):
        # normal_y_mode = 0: normalize Y by combing all dimension.
        # normal_y_mode = 1: normalize Y by each dimension.
        super(cigp, self).__init__()

        #normalize X independently for each dimension
        self.Xmean = X.mean(0)
        self.Xstd = X.std(0)
        self.X = (X - self.Xmean.expand_as(X)) / (self.Xstd.expand_as(X) + EPS)

        if normal_y_mode == 0:
            # normalize y all together
            self.Ymean = Y.mean()
            self.Ystd = Y.std()
            self.Y = (Y - self.Ymean.expand_as(Y)) / (self.Ystd.expand_as(Y) + EPS)
        elif normal_y_mode == 1:
        # option 2: normalize y by each dimension
            self.Ymean = Y.mean(0)
            self.Ystd = Y.std(0)
            self.Y = (Y - self.Ymean.expand_as(Y)) / (self.Ystd.expand_as(Y) + EPS)

        # GP hyperparameters
        self.log_beta = nn.Parameter(torch.ones(1) * 0)   # a large noise by default. Smaller value makes larger noise variance.
        self.log_length_scale = nn.Parameter(torch.zeros(X.size(1)))    # ARD length scale
        self.log_scale = nn.Parameter(torch.zeros(1))   # kernel scale
        
        # mean function
        if meanFunc == 0:
            self.meanFunc = zeroFunc(Y.size(1))
        self.meanFunc = meanFunc
                
    # define kernel function
    def kernel(self, X1, X2):
        # the common RBF kernel
        X1 = X1 / self.log_length_scale.exp()
        X2 = X2 / self.log_length_scale.exp()
        # X1_norm2 = X1 * X1
        # X2_norm2 = X2 * X2
        X1_norm2 = torch.sum(X1 * X1, dim=1).view(-1, 1)
        X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)

        K = -2.0 * X1 @ X2.t() + X1_norm2.expand(X1.size(0), X2.size(0)) + X2_norm2.t().expand(X1.size(0), X2.size(0))  #this is the effective Euclidean distance matrix between X1 and X2.
        K = self.log_scale.exp() * torch.exp(-0.5 * K)
        return K

    def kernel_matern3(self, x1, x2):
        """
        latex formula:
        \sigma ^2\left( 1+\frac{\sqrt{3}d}{\rho} \right) \exp \left( -\frac{\sqrt{3}d}{\rho} \right)
        :param x1: x_point1
        :param x2: x_point2
        :return: kernel matrix
        """
        const_sqrt_3 = torch.sqrt(torch.ones(1) * 3)
        x1 = x1 / self.log_length_matern3.exp()
        x2 = x2 / self.log_length_matern3.exp()
        distance = const_sqrt_3 * torch.cdist(x1, x2, p=2)
        k_matern3 = self.log_coe_matern3.exp() * (1 + distance) * (- distance).exp()
        return k_matern3

    def kernel_matern5(self, x1, x2):
        """
        latex formula:
        \sigma ^2\left( 1+\frac{\sqrt{5}}{l}+\frac{5r^2}{3l^2} \right) \exp \left( -\frac{\sqrt{5}distance}{l} \right)
        :param x1: x_point1
        :param x2: x_point2
        :return: kernel matrix
        """
        const_sqrt_5 = torch.sqrt(torch.ones(1) * 5)
        x1 = x1 / self.log_length_matern5.exp()
        x2 = x2 / self.log_length_matern5.exp()
        distance = const_sqrt_5 * torch.cdist(x1, x2, p=2)
        k_matern5 = self.log_coe_matern5.exp() * (1 + distance + distance ** 2 / 3) * (- distance).exp()
        return k_matern5


    def forward(self, Xte):
        n_test = Xte.size(0)
        Xte = ( Xte - self.Xmean.expand_as(Xte) ) / self.Xstd.expand_as(Xte)

        Sigma = self.kernel(self.X, self.X) + self.log_beta.exp().pow(-1) * torch.eye(self.X.size(0)) \
            + JITTER * torch.eye(self.X.size(0))

        kx = self.kernel(self.X, Xte)
        L = torch.cholesky(Sigma)
        LinvKx,_ = torch.triangular_solve(kx, L, upper = False)

        # option 1
        mean = self.meanFunc(Xte)
        # mean = mean + kx.t() @ torch.cholesky_solve(self.Y, L)  # torch.linalg.cholesky() #this is Wrong implementation that forgets to minus the mean.
        mean = mean + kx.t() @ torch.cholesky_solve(self.Y-self.meanFunc(self.X), L)  # torch.linalg.cholesky()
        
        var_diag = self.kernel(Xte, Xte).diag().view(-1, 1) \
            - (LinvKx**2).sum(dim = 0).view(-1, 1)

        # add the noise uncertainty
        var_diag = var_diag + self.log_beta.exp().pow(-1)

        # de-normalized
        mean = mean * self.Ystd.expand_as(mean) + self.Ymean.expand_as(mean)
        var_diag = var_diag.expand_as(mean) * self.Ystd**2

        return mean, var_diag


    def negative_log_likelihood(self):
        y_num, y_dimension = self.Y.shape
        Sigma = self.kernel(self.X, self.X) + self.log_beta.exp().pow(-1) * torch.eye(
            self.X.size(0)) + JITTER * torch.eye(self.X.size(0))

        L = torch.linalg.cholesky(Sigma)
        #option 1 (use this if torch supports)
        Gamma,_ = torch.triangular_solve(self.Y - self.meanFunc(self.X), L, upper = False)
        #option 2
        # gamma = L.inverse() @ Y       # we can use this as an alternative because L is a lower triangular matrix.

        nll =  0.5 * (Gamma ** 2).sum() +  L.diag().log().sum() * y_dimension  \
            + 0.5 * y_num * torch.log(2 * torch.tensor(PI)) * y_dimension
        return nll

    def train_adam(self, niteration=10, lr=0.1):
        # adam optimizer
        # uncommont the following to enable
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        for i in range(niteration):
            optimizer.zero_grad()
            # self.update()
            loss = self.negative_log_likelihood()
            loss.backward()
            optimizer.step()
            # print('loss_nll:', loss.item())
            # print('iter', i, ' nll:', loss.item())
            print('iter', i, 'nll:{:.5f}'.format(loss.item()))


    def train_bfgs(self, niteration=50, lr=0.1):
        # LBFGS optimizer
        # Some optimization algorithms such as Conjugate Gradient and LBFGS need to reevaluate the function multiple times, so you have to pass in a closure that allows them to recompute your model. The closure should clear the gradients, compute the loss, and return it.
        optimizer = torch.optim.LBFGS(self.parameters(), lr=lr)  # lr is very important, lr>0.1 lead to failure
        for i in range(niteration):
            # optimizer.zero_grad()
            # LBFGS
            def closure():
                optimizer.zero_grad()
                # self.update()
                loss = self.negative_log_likelihood()
                loss.backward()
                # print('nll:', loss.item())
                # print('iter', i, ' nll:', loss.item())
                print('iter', i, 'nll:{:.5f}'.format(loss.item()))
                return loss

            # optimizer.zero_grad()
            optimizer.step(closure)
        # print('loss:', loss.item())

    # TODO: add conjugate gradient method

# %%
if __name__ == "__main__":
    print('testing')
    print(torch.__version__)

    # single output test 1
    xte = torch.linspace(0, 6, 100).view(-1, 1)
    yte = torch.sin(xte) + 10

    xtr = torch.rand(16, 1) * 6
    ytr = torch.sin(xtr) + torch.randn(16, 1) * 0.5 + 10

    meanfunction = linearRegression(xtr.size(1), ytr.size(1))
    model = cigp(xtr, ytr, meanfunction)
    model.train_adam(200, lr=0.1)
    # model.train_bfgs(50, lr=0.1)

    with torch.no_grad():
        ypred, ypred_var = model(xte)

    plt.errorbar(xte, ypred.reshape(-1).detach(), ypred_var.sqrt().squeeze().detach(), fmt='r-.' ,alpha = 0.2)
    plt.plot(xtr, ytr, 'b+')
    plt.show()

    # single output test 2
    xte = torch.rand(128,2) * 2
    yte = torch.sin(xte.sum(1)).view(-1,1) + 10

    xtr = torch.rand(32, 2) * 2
    ytr = torch.sin(xtr.sum(1)).view(-1,1) + torch.randn(32, 1) * 0.5 + 10

    meanfunction = linearRegression(xtr.size(1), ytr.size(1))
    model = cigp(xtr, ytr, meanfunction)
    # model = cigp(xtr, ytr)
    model.train_adam(300, lr=0.1)
    # model.train_bfgs(50, lr=0.01)

    with torch.no_grad():
        ypred, ypred_var = model(xte)

    # plt.errorbar(xte.sum(1), ypred.reshape(-1).detach(), ystd.sqrt().squeeze().detach(), fmt='r-.' ,alpha = 0.2)
    plt.plot(xte.sum(1), yte, 'b+')
    plt.plot(xte.sum(1), ypred.reshape(-1).detach(), 'r+')
    # plt.plot(xtr.sum(1), ytr, 'b+')
    plt.show()
    

    # multi output test
    xte = torch.linspace(0, 6, 100).view(-1, 1)
    yte = torch.hstack([torch.sin(xte),
                       torch.cos(xte),
                        xte.tanh()] )

    xtr = torch.rand(32, 1) * 6
    ytr = torch.sin(xtr) + torch.rand(32, 1) * 0.5
    ytr = torch.hstack([torch.sin(xtr),
                       torch.cos(xtr),
                        xtr.tanh()] )+ torch.randn(32, 3) * 0.2

    meanfunction = linearRegression(xtr.size(1), ytr.size(1))
    model = cigp(xtr, ytr, meanfunction, 1)
    # model = cigp(xtr, ytr, 1)
    model.train_adam(100, lr=0.1)
    # model.train_bfgs(50, lr=0.001)

    with torch.no_grad():
        ypred, ypred_var = model(xte)

    # plt.errorbar(xte, ypred.detach(), ypred_var.sqrt().squeeze().detach(),fmt='r-.' ,alpha = 0.2)
    plt.plot(xte, ypred.detach(),'r-.')
    plt.plot(xtr, ytr, 'b+')
    plt.plot(xte, yte, 'k-')
    plt.show()

    # plt.close('all')
    plt.plot(xtr, ytr, 'b+')
    for i in range(3):
        plt.plot(xte, yte[:, i], label='truth', color='r')
        plt.plot(xte, ypred[:, i], label='prediction', color='navy')
        plt.fill_between(xte.squeeze(-1).detach().numpy(),
                         ypred[:, i].squeeze(-1).detach().numpy() + torch.sqrt(ypred_var[:, i].squeeze(-1)).detach().numpy(),
                         ypred[:, i].squeeze(-1).detach().numpy() - torch.sqrt(ypred_var[:, i].squeeze(-1)).detach().numpy(),
                         alpha=0.2)
    plt.show()

# %%
