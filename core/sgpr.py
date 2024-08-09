#Author: Zidong Chen
#Date: 2024-06-22
# This is the implementation of the variational sparse Gaussian process (VSGP) model.
# More details can be found in the paper "Variational Learning of Inducing Variables in Sparse Gaussian Processes" by Titsias (2009).

import torch
import torch.nn as nn
import core.GP_CommonCalculation as GP
from core.kernel import ARDKernel

print(torch.__version__)
# I use torch (1.11.0) for this work. lower version may not work.

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Fixing strange error if run in MacOS
JITTER = 1e-3
EPS = 1e-10
PI = 3.1415
torch.set_default_dtype(torch.float64)

class vsgp(nn.Module):
    def __init__(self, X, Y, num_inducing):
        super(vsgp, self).__init__()

        self.data = GP.DataNormalization(method='standard')
        self.data.fit(X, 'x')
        self.data.fit(Y, 'y')
        self.X = self.data.normalize(X, 'x')
        self.Y = self.data.normalize(Y, 'y')

        # GP hyperparameters
        self.log_beta = nn.Parameter(torch.ones(1) * 0)  # Initial noise level
        self.device = self.X.device

        # Inducing points
        input_dim=self.X.size(1)
        self.kernel = ARDKernel(input_dim)
        self.xm = nn.Parameter(torch.rand((num_inducing, input_dim)))  # Inducing points

    def negative_log_likelihood(self):
        """Negative lower bound as the loss function to minimize."""
        n = self.X.size(0)
        K_mm = self.kernel(self.xm, self.xm) + JITTER * torch.eye(self.xm.size(0), device=self.device)
        L = torch.linalg.cholesky(K_mm)
        K_mn = self.kernel(self.xm, self.X)
        K_nn = self.kernel(self.X, self.X)
        A = torch.linalg.solve_triangular(L, K_mn, upper=False)
        A = A * torch.sqrt(self.log_beta.exp())
        AAT = A @ A.t()
        B = AAT + (1+JITTER) * torch.eye(self.xm.size(0),device=self.device)
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
        K_mm = self.kernel(self.xm, self.xm) + JITTER * torch.eye(self.xm.size(0),device=self.device)
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
        Xte = self.data.normalize(Xte,'x')

        K_tt = self.kernel(Xte, Xte)
        K_tm = self.kernel(Xte, self.xm)
        K_mt = K_tm.t()
        mean_m, A_m, K_mm_inv = self.optimal_inducing_point()
        mean = (K_tm @ K_mm_inv) @ mean_m
        var = (K_tt - K_tm @ K_mm_inv @ K_mt +
               K_tm @ K_mm_inv @ A_m @ K_mm_inv @ K_mt)
        var_diag = var.diag().view(-1, 1)
        # de-normalized
        mean = self.data.denormalize(mean, 'y')
        var_diag = self.data.denormalize_cov(var_diag, 'y')

        return mean, var_diag

    def train_adam(self, niteration=10, lr=0.1):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        for i in range(niteration):
            optimizer.zero_grad()
            loss = self.negative_log_likelihood()
            loss.backward()
            optimizer.step()
            #print('iter', i, ' nll:', loss.item())
        print('done2')
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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import data_sample.generate_example_data as data
    print('testing')
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xtr, ytr, xte, yte = data.generate(2000, 500, seed=2)
    xtr = xtr.to(device)
    ytr = ytr.to(device)
    xte = xte.to(device)
    yte = yte.to(device)

    #print(ytr)
    GPmodel = vsgp(xtr, ytr, num_inducing=200).to(device)
    optimizer = torch.optim.Adam(GPmodel.parameters(), lr=0.1)

    import time

    iteration_times = []
    for i in range(200):
        start_time = time.time()
        optimizer.zero_grad()
        loss = GPmodel.negative_log_likelihood()
        loss.backward()
        optimizer.step()
        end_time= time.time()
        iteration_times.append(end_time - start_time)

    average_iteration_time = sum(iteration_times) / len(iteration_times)


    with torch.no_grad():
        ypred, ypred_var = GPmodel.forward(xte)
        mse= torch.mean((yte-ypred)**2)

        R_square = 1 - torch.sum((yte - ypred) ** 2) / torch.sum((yte - yte.mean()) ** 2)
        print(average_iteration_time)
        print(mse,R_square)
        plt.plot(xte.cpu().numpy(), yte.cpu().numpy(), 'r.')
        plt.plot(xte.cpu().numpy(), ypred.cpu().numpy(), 'b.')
        plt.show()