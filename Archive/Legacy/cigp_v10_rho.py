# Conditional independent Gaussian process for vector output regression based on pytorch
# v10: A stable version. improve over the v02 version to fix nll bug; adapt to torch 1.11.0.
#
# Author: Wei W. Xing (wxing.me)
# Email: wayne.xingle@gmail.com
# Date: 2022-03-23


# %%
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import os

print("cigp_mean torch version:", torch.__version__)
# I use torch (1.11.0) for this work. lower version may not work.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Fixing strange error if run in MacOS
JITTER = 1e-6
EPS = 1e-10
PI = 3.1415


class ConstMeanCIGP(nn.Module):
    def __init__(self, X, Y, yln, yhn, normal_mode=1):
        super(ConstMeanCIGP, self).__init__()
        self.X = X
        self.Y = Y

        # normalize X independently for each dimension
        self.Xmean = X.mean(0)
        self.Xstd = X.std(0)
        self.X = (X - self.Xmean.expand_as(X)) / (self.Xstd.expand_as(X) + EPS)

        # normalize y all together
        self.Ymean = Y.mean()
        self.Ystd = Y.std()
        self.Y = (Y - self.Ymean.expand_as(Y)) / self.Ystd.expand_as(Y)

        # option 2: normalize y by each dimension
        # self.Ymean = Y.mean(0)
        # self.Ystd = Y.std(0)
        # self.Y = (Y - self.Ymean.expand_as(Y)) / (self.Ystd.expand_as(Y) + EPS)

        self.log_beta = nn.Parameter(torch.ones(1) * 0)  # a large noise
        self.log_length_scale = nn.Parameter(torch.zeros(X.size(1)))
        self.log_scale = nn.Parameter(torch.zeros(1))
        # self.rho = nn.Parameter(torch.ones(1))
        self.rho = nn.Parameter(torch.ones(1))

    # define kernel function
    def kernel(self, X, X2):
        length_scale = torch.exp(self.log_length_scale).view(1, -1)

        X = X / length_scale.expand(X.size(0), length_scale.size(1))
        X2 = X2 / length_scale.expand(X2.size(0), length_scale.size(1))

        X_norm2 = torch.sum(X * X, dim=1).view(-1, 1)
        X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)

        K = -2.0 * X @ X2.t() + X_norm2.expand(X.size(0), X2.size(0)) + X2_norm2.t().expand(X.size(0), X2.size(0))
        K = self.log_scale.exp() * torch.exp(-0.5 * K)

        # X1 = X1 / self.log_length_scale.exp() ** 2
        # X2 = X2 / self.log_length_scale.exp() ** 2
        # X1_norm2 = X1 * X1
        # X2_norm2 = X2 * X2

        # K = -2.0 * X1 @ X2.t() + X1_norm2.expand(X1.size(0), X2.size(0)) + X2_norm2.t().expand(X1.size(0), X2.size(
        #     0))  # this is the effective Euclidean distance matrix between X1 and X2.
        # K = self.log_scale.exp() * torch.exp(-0.5 * K)
        return K

    def forward(self, Xte, ytr_m, ytr_v, yte_m, yte_v):
        n_test = Xte.size(0)

        Xte = ( Xte - self.Xmean.expand_as(Xte) ) / (self.Xstd.expand_as(Xte) + EPS)
        ytr_m = (ytr_m - self.Ymean.expand_as(ytr_m)) / (self.Ystd.expand_as(ytr_m) + EPS)
        yte_m = (yte_m - self.Ymean.expand_as(yte_m)) / (self.Ystd.expand_as(yte_m) + EPS)
        
        Sigma = self.kernel(self.X, self.X) + self.log_beta.exp().pow(-1) * torch.eye(self.X.size(0)) \
                + JITTER * torch.eye(self.X.size(0))

        kx = self.kernel(self.X, Xte)
        L = torch.cholesky(Sigma)
        LinvKx, _ = torch.triangular_solve(kx, L, upper=False)

        # option 1
        mean = kx.t() @ torch.cholesky_solve(self.Y - ytr_m * self.rho, L)  # torch.linalg.cholesky()
        var_diag = self.log_scale.exp().expand(n_test, 1) \
                   - (LinvKx ** 2).sum(dim=0).view(-1, 1)

        # add the noise uncertainty
        var_diag = var_diag + self.log_beta.exp().pow(-1)
        # v = var_diag.shape
        
        mean = mean + yte_m * self.rho
        mean = mean * self.Ystd.expand_as(mean) + self.Ymean.expand_as(mean)

        # wtw = (self.rho.T @ self.rho).diag()
        # var_list = [ yte_v[:,0][i] * wtw + var_diag[i] * torch.ones(wtw.shape[0]) for i in range(len(yte_v[:,0]))]
        # var_diag = torch.stack(var_list)

        # var_diag = var_diag * self.Ystd ** 2

        return mean

    def negative_log_likelihood(self, ytr_m):
        ytr_m = (ytr_m - self.Ymean.expand_as(ytr_m)) / (self.Ystd.expand_as(ytr_m) + EPS)
        
        y_num, y_dimension = self.Y.shape
        Sigma = self.kernel(self.X, self.X) + self.log_beta.exp().pow(-1) * torch.eye(
            self.X.size(0)) + JITTER * torch.eye(self.X.size(0))

        L = torch.linalg.cholesky(Sigma)
        # option 1 (use this if torch supports)
        # Gamma, _ = torch.triangular_solve(self.Y - ytr_m @ self.rho, L, upper=False)
       
        # option 2
        gamma = L.inverse() @ (self.Y - ytr_m * self.rho)     # we can use this as an alternative because L is a lower triangular matrix.

        nll = 0.5 * (gamma ** 2).sum() + L.diag().log().sum() * y_dimension \
              + 0.5 * y_num * torch.log(2 * torch.tensor(PI)) * y_dimension
        return nll

    def train_adam(self,ytr_m, niteration=10, lr=0.1):
        # adam optimizer
        # uncommont the following to enable
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        for i in range(niteration):
            optimizer.zero_grad()
            # self.update()
            loss = self.negative_log_likelihood(ytr_m)
            loss.backward()
            optimizer.step()
            # print('loss_nnl:', loss.item())
            # print('iter', i, ' nnl:', loss.item())
            print('iter', i, 'nll:{:.5f}'.format(loss.item()))
            # print("rho:", i, self.rho)