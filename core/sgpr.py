# Author: Zidong Chen
# Date: 2024-06-22
# This is the implementation of the variational sparse Gaussian process (VSGP) model.
# More details can be found in the paper "Variational Learning of Inducing Variables in Sparse Gaussian Processes" by Titsias (2009).

import torch
import torch.nn as nn
import core.GP_CommonCalculation as GP
from core.kernel import ARDKernel
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Fixing strange error if run in MacOS
JITTER = 1e-3
EPS = 1e-10
PI = 3.1415
torch.set_default_dtype(torch.float64)


class vsgp(nn.Module):
    def __init__(self, kernel, num_inducing, input_dim, log_beta=None, device='cpu'):
        super(vsgp, self).__init__()
        # GP hyperparameters
        if log_beta is None:
            self.log_beta = nn.Parameter(torch.ones(1) * 0)
        else:
            self.log_beta = nn.Parameter(log_beta)

        self.device = device

        # Inducing points
        self.kernel = kernel
        self.xm = nn.Parameter(torch.rand((num_inducing, input_dim), dtype=torch.float64).to(device))  # Inducing points

    def negative_log_likelihood(self, X, Y):
        """Negative lower bound as the loss function to minimize."""
        n = X.size(0)
        K_mm = self.kernel(self.xm, self.xm) + JITTER * torch.eye(self.xm.size(0), device=self.device)
        L = torch.linalg.cholesky(K_mm)
        K_mn = self.kernel(self.xm, X)
        K_nn = self.kernel(X, X)
        A = torch.linalg.solve_triangular(L, K_mn, upper=False)
        A = A * torch.sqrt(self.log_beta.exp())
        AAT = A @ A.t()
        B = AAT + (1 + JITTER) * torch.eye(self.xm.size(0), device=self.device)
        LB = torch.linalg.cholesky(B)

        c = torch.linalg.solve_triangular(LB, A @ Y, upper=False)
        c = c * torch.sqrt(self.log_beta.exp())
        nll = (n / 2 * torch.log(2 * torch.tensor(PI)) +
               torch.sum(torch.log(torch.diagonal(LB))) +
               n / 2 * torch.log(1 / self.log_beta.exp()) +
               self.log_beta.exp() / 2 * torch.sum(Y * Y) -
               0.5 * torch.sum(c.squeeze() * c.squeeze()) +
               self.log_beta.exp() / 2 * torch.sum(torch.diagonal(K_nn)) -
               0.5 * torch.trace(AAT))
        return nll

    def optimal_inducing_point(self, X, Y):
        """Compute optimal inducing points mean and covariance."""
        K_mm = self.kernel(self.xm, self.xm) + JITTER * torch.eye(self.xm.size(0), device=self.device)
        L = torch.linalg.cholesky(K_mm)
        L_inv = torch.inverse(L)
        K_mm_inv = L_inv.t() @ L_inv

        K_mn = self.kernel(self.xm, X)
        K_nm = K_mn.t()
        sigma = torch.inverse(K_mm + self.log_beta.exp() * K_mn @ K_nm)

        mean_m = self.log_beta.exp() * (K_mm @ sigma @ K_mn) @ Y
        A_m = K_mm @ sigma @ K_mm
        return mean_m, A_m, K_mm_inv

    def forward(self, X, Y, Xte_normalized):
        """Compute mean and variance for posterior distribution."""

        K_tt = self.kernel(Xte_normalized, Xte_normalized)
        K_tm = self.kernel(Xte_normalized, self.xm)
        K_mt = K_tm.t()
        mean_m, A_m, K_mm_inv = self.optimal_inducing_point(X, Y)
        mean = (K_tm @ K_mm_inv) @ mean_m
        var = (K_tt - K_tm @ K_mm_inv @ K_mt +
               K_tm @ K_mm_inv @ A_m @ K_mm_inv @ K_mt)
        var_diag = var.diag().view(-1, 1) + self.log_beta.exp().pow(-1)

        return mean, var_diag

    def train_adam(self, X, Y, niteration=10, lr=0.1):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        for i in range(niteration):
            optimizer.zero_grad()
            loss = self.negative_log_likelihood(X, Y)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f'Iteration {i}: Negative Log-Likelihood = {loss.item():.5f}')
        print('Training completed.')

    def train_lbfgs(self, X, Y, max_iter=20, lr=0.3):
        """Train model using LBFGS optimizer."""
        optimizer = torch.optim.LBFGS(self.parameters(), max_iter=max_iter, lr=lr)

        def closure():
            optimizer.zero_grad()
            loss = self.negative_log_likelihood(X, Y)
            loss.backward()  # Retain the graph
            print(f'Loss: {loss.item():.5f}')
            return loss

        optimizer.step(closure)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import data_sample.generate_example_data as data

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xtr, ytr, xte, yte = data.generate(2000, 500, seed=2)
    xtr = xtr.to(device)
    ytr = ytr.to(device)
    xte = xte.to(device)
    yte = yte.to(device)

    # Perform normalization outside the model
    normalizer = GP.DataNormalization(method='standard')
    normalizer.fit(xtr, 'x')
    normalizer.fit(ytr, 'y')
    xtr_normalized = normalizer.normalize(xtr, 'x')
    ytr_normalized = normalizer.normalize(ytr, 'y')
    xte_normalized = normalizer.normalize(xte, 'x')

    # Create the model
    GPmodel = vsgp(kernel=ARDKernel(1), num_inducing=200, input_dim=xtr_normalized.size(1), device=device).to(device)

    # Training using Adam optimizer
    GPmodel.train_adam(xtr_normalized, ytr_normalized, niteration=200, lr=0.1)

    # Evaluate the model on the test set
    GPmodel.eval()
    with torch.no_grad():
        ypred, ypred_var = GPmodel.forward(xtr_normalized, ytr_normalized, xte_normalized)
        ypred = normalizer.denormalize(ypred, 'y')
        ypred_var = normalizer.denormalize_cov(ypred_var, 'y')
        mse = torch.mean((yte - ypred) ** 2)

        R_square = 1 - torch.sum((yte - ypred) ** 2) / torch.sum((yte - yte.mean()) ** 2)
        print(f'MSE: {mse.item():.5f}, R²: {R_square.item():.5f}')
        plt.plot(xte.cpu().numpy(), yte.cpu().numpy(), 'r.')
        plt.plot(xte.cpu().numpy(), ypred.cpu().numpy(), 'b.')
        plt.show()
