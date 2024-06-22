#Author: Zidong Chen
#Date: 2024-06-22
# This is the implementation of the variational sparse Gaussian process (VSGP) model.
# More details can be found in the paper "Variational Learning of Inducing Variables in Sparse Gaussian Processes" by Titsias (2009).

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from data_process import data_process
print(torch.__version__)
# I use torch (1.11.0) for this work. lower version may not work.

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Fixing strange error if run in MacOS
JITTER = 1e-3
EPS = 1e-10
PI = 3.1415



class vsgp(nn.Module):
    def __init__(self, X, Y, num_inducing, normal_y_mode=0):
        super(vsgp, self).__init__()

        self.data = data_process(X, Y, normal_y_mode)
        self.X, self.Y = self.data.normalize(X, Y)
        self.X, self.Y = self.data.remove(self.X, self.Y)

        # GP hyperparameters
        self.log_beta = nn.Parameter(torch.ones(1) * -4)  # Initial noise level
        self.log_length_scale = nn.Parameter(torch.zeros(X.size(1)))  # ARD length scale
        self.log_scale = nn.Parameter(torch.zeros(1))  # Kernel scale

        # Inducing points
        subset_indices = torch.randperm(self.X.size(0))[:num_inducing]
        self.xm = nn.Parameter(self.X[subset_indices])  # Inducing points

    def kernel(self, X1, X2):
        """Common RBF kernel."""
        X1 = X1 / self.log_length_scale.exp()
        X2 = X2 / self.log_length_scale.exp()
        X1_norm2 = torch.sum(X1 * X1, dim=1).view(-1, 1)
        X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)

        K = -2.0 * X1 @ X2.t() + X1_norm2.expand(X1.size(0), X2.size(0)) + X2_norm2.t().expand(X1.size(0), X2.size(0))
        K = self.log_scale.exp() * torch.exp(-0.5 * K)
        return K

    def negative_lower_bound(self):
        """Negative lower bound as the loss function to minimize."""
        n = self.X.size(0)
        K_mm = self.kernel(self.xm, self.xm) + JITTER * torch.eye(self.xm.size(0))
        L = torch.linalg.cholesky(K_mm)
        K_mn = self.kernel(self.xm, self.X)
        K_nn = self.kernel(self.X, self.X)
        A = torch.linalg.solve_triangular(L, K_mn, upper=False)
        A = A * torch.sqrt(self.log_beta.exp())
        AAT = A @ A.t()
        B = torch.eye(self.xm.size(0)) + AAT + JITTER * torch.eye(self.xm.size(0))
        LB = torch.linalg.cholesky(B)

        c = torch.linalg.solve_triangular(LB, A @ self.Y, upper=False)
        c = c * torch.sqrt(self.log_beta.exp())
        nll = (n / 2 * torch.log(2 * torch.tensor(PI)) +
               torch.sum(torch.log(torch.diagonal(LB))) +
               n / 2 * torch.log(1 / self.log_beta.exp()) +
               self.log_beta.exp() / 2 * torch.sum(self.Y * self.Y) -
               0.5 * torch.sum(c.squeeze() * c.squeeze()) +
               self.log_beta.exp() / 2 * torch.sum(torch.diagonal(K_nn)) -
               0.5 * torch.trace(AAT))
        return nll

    def optimal_inducing_point(self):
        """Compute optimal inducing points mean and covariance."""
        K_mm = self.kernel(self.xm, self.xm) + JITTER * torch.eye(self.xm.size(0))
        L = torch.linalg.cholesky(K_mm)
        L_inv = torch.inverse(L)
        K_mm_inv = L_inv.t() @ L_inv

        K_mn = self.kernel(self.xm, self.X)
        K_nm = K_mn.t()
        sigma = torch.inverse(K_mm + self.log_beta.exp() * K_mn @ K_nm)

        mean_m = self.log_beta.exp() * (K_mm @ sigma @ K_mn) @ self.Y
        A_m = K_mm @ sigma @ K_mm
        return mean_m, A_m, K_mm_inv

    def forward(self, Xte):
        """Compute mean and variance for posterior distribution."""
        Xte = self.data.normalize(Xte)
        K_tt = self.kernel(Xte, Xte)
        K_tm = self.kernel(Xte, self.xm)
        K_mt = K_tm.t()
        mean_m, A_m, K_mm_inv = self.optimal_inducing_point()
        mean = (K_tm @ K_mm_inv) @ mean_m
        var = (K_tt - K_tm @ K_mm_inv @ K_mt +
               K_tm @ K_mm_inv @ A_m @ K_mm_inv @ K_mt)
        var_diag = var.diag().view(-1, 1)
        mean, var_diag = self.data.denormalize_result(mean, var_diag)
        return mean, var_diag

    def train_adam(self, niteration=10, lr=0.1):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        for i in range(niteration):
            optimizer.zero_grad()
            loss = self.negative_lower_bound()
            loss.backward()
            optimizer.step()
            print('iter', i, ' nll:', loss.item())

    def train_lbfgs(self, max_iter=20,lr=0.3):
        """Train model using LBFGS optimizer."""
        optimizer = torch.optim.LBFGS(self.parameters(), max_iter=max_iter,lr=lr)

        def closure():
            optimizer.zero_grad()
            loss = self.negative_lower_bound()
            loss.backward()  # Retain the graph
            print(self.log_length_scale.exp().item(), self.log_scale.exp().item())
            print(loss)
            return loss

        optimizer.step(closure)

# Test Script
if __name__ == "__main__":
    print('testing')
    print(torch.__version__)

    # single output test 1
    xte = torch.linspace(0, 6, 100).view(-1, 1)
    yte = torch.sin(xte) + 10

    xtr = torch.rand(16, 1) * 6
    ytr = torch.sin(xtr) + torch.randn(16, 1) * 0.5 + 10

    model = vsgp(xtr, ytr,10)
    #model.train_adam(200, lr=0.1)
    model.train_lbfgs(20, lr=0.1)

    with torch.no_grad():
        ypred, ypred_var = model.forward(xte)

    plt.errorbar(xte, ypred.reshape(-1).detach(), ypred_var.sqrt().squeeze().detach(), fmt='r-.' ,alpha = 0.2)
    plt.plot(xtr, ytr, 'b+')
    plt.show()