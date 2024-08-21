import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import core.GP_CommonCalculation as GP

print(torch.__version__)
# I use torch (1.11.0) for this work. lower version may not work.
from core.kernel import ARDKernel
import os
from core.cigp import cigp

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Fixing strange error if run in MacOS
JITTER = 1e-6
EPS = 1e-10
PI = 3.1415


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import GP_CommonCalculation as GP

JITTER = 1e-6
EPS = 1e-10
PI = 3.1415
torch.set_default_dtype(torch.float64)

class inputwarp_gp(nn.Module):
    def __init__(self, kernel=ARDKernel(1), log_beta=None, device='cpu'):
        super(inputwarp_gp, self).__init__()

        # GP hyperparameters
        if log_beta is None:
            self.log_beta = nn.Parameter(torch.ones(1) * -4)
        else:
            self.log_beta = nn.Parameter(log_beta)
        self.kernel = kernel
        self.device = device
        self.warp = GP.Warp(method='kumar', initial_a=1.0, initial_b=1.0)

    def forward(self, Xtr, Ytr, Xte):
        # Apply warp transformation
        Xtr = self.warp.transform(Xtr)
        Xte = self.warp.transform(Xte)

        # GP calculation
        Sigma = self.kernel(Xtr, Xtr) + self.log_beta.exp().pow(-1) * torch.eye(Xtr.size(0), device=self.device) + JITTER * torch.eye(Xtr.size(0), device=self.device)
        kx = self.kernel(Xtr, Xte)
        L = torch.linalg.cholesky(Sigma)
        LinvKx = torch.linalg.solve_triangular(L, kx, upper=False)

        mean = kx.t() @ torch.cholesky_solve(Ytr, L)
        var_diag = self.kernel(Xte, Xte).diag().view(-1, 1) - (LinvKx ** 2).sum(dim=0).view(-1, 1)
        var_diag = var_diag + self.log_beta.exp().pow(-1)

        return mean, var_diag

    def negative_log_likelihood(self, Xtr, Ytr):
        Xtr = self.warp.transform(Xtr)
        Sigma = self.kernel(Xtr, Xtr) + self.log_beta.exp().pow(-1) * torch.eye(Xtr.size(0), device=self.device) + JITTER * torch.eye(Xtr.size(0), device=self.device)
        L = torch.linalg.cholesky(Sigma)
        Gamma = torch.linalg.solve_triangular(L, Ytr, upper=False)

        y_num, y_dimension = Ytr.shape
        nll = 0.5 * (Gamma ** 2).sum() + L.diag().log().sum() * y_dimension + 0.5 * y_num * torch.log(2 * torch.tensor(PI)) * y_dimension
        return nll

    def train_adam(self, Xtr, Ytr, niteration=10, lr=0.1):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for i in range(niteration):
            optimizer.zero_grad()
            loss = self.negative_log_likelihood(Xtr, Ytr)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('iter', i, 'nll:{:.5f}'.format(loss.item()))

# %%
if __name__ == '__main__':
    # Example usage
    np.random.seed(0)
    torch.manual_seed(0)
    n = 500
    X = torch.rand(n,1)*5

    X_warped = 2 * torch.exp(X + 1) / (1 + torch.exp(X + 1))
    Y = 2 * X_warped.sum(dim=1) + 0.5 * torch.normal(mean=0, std=0.1, size=(n,))
    Y=Y.view(-1,1)
    Xte = torch.linspace(0, 5, n).reshape(n, 1)
    Xte_warped = 2 * torch.exp(Xte + 1) / (1 + torch.exp(Xte + 1))
    Yte = 2 * Xte_warped.sum(dim=1)
    Yte=Yte.view(-1,1)

    # Perform normalization outside the model
    normalizer = GP.DataNormalization(method='min_max')
    normalizer.fit(X, 'x')
    normalizer.fit(Y, 'y')
    normalizer.fit(Xte, 'xte')
    X_normalized = normalizer.normalize(X, 'x')
    Y_normalized = normalizer.normalize(Y, 'y')
    Xte_normalized= normalizer.normalize(Xte, 'xte')

    # Create the models
    model = inputwarp_gp()
    model2 = cigp()

    # Train the models
    model.train_adam(X_normalized, Y_normalized, niteration=100, lr=0.1)
    model2.train_adam(X_normalized, Y_normalized, niteration=100, lr=0.1)

    # Test the models

    mean, var = model.forward(X_normalized, Y_normalized, Xte_normalized)
    mean2, var2 = model2.forward(X_normalized, Y_normalized,Xte_normalized)

    # De-normalize the predictions
    mean = normalizer.denormalize(mean, 'y')
    var = normalizer.denormalize_cov(var, 'y')

    # Plot the results
    plt.figure()
    plt.scatter(X, Y)
    plt.plot(Xte.numpy(), mean.detach().numpy(), 'r')
    print(model.warp.a, model.warp.b)

    plt.show()

    mse = torch.mean((mean - Yte) ** 2)
    mse2 = torch.mean((mean2 - Yte) ** 2)
    print(f'Test warped MSE: {mse.item()}', f'Test cigp MSE: {mse2.item()}')
    mse2 = torch.mean((mean2 - Yte) ** 2)
    print(f'Test MSE: {mse2.item()}')
    R_square = 1 - torch.sum((mean - Yte) ** 2) / torch.sum((Yte - torch.mean(Yte)) ** 2)
    print('inputWarp R square:', R_square.item())
    R_square2 = 1 - torch.sum((mean2 - Yte) ** 2) / torch.sum((Yte - torch.mean(Yte)) ** 2)
    print('cigp R square2:', R_square2.item())
