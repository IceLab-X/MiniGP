# ---- coding: utf-8 ----
# @author: Xing Wei
# @version: v14, demonstration of mixed kernel (Linear+matern3+matern5), 
# @license: (C) Copyright 2021, AMML Group Limited.

"""
CIGP, GRP torch model using nn.module
fixed beta
NOTE THIS:
this version uses `Matern3 Kernel`
"""
import os
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Fixing strange error if run in MacOS
JITTER = 1e-6
EPS = 1e-10
PI = 3.1415


class CIGP(nn.Module):

    def __init__(
            self,
            x,
            y,
            normal_y_mode=0,
            **kwargs
    ):
        # normal_y_mode = 0: normalize y by combing all dimension.
        # normal_y_mode = 1: normalize y by each dimension.
        super(CIGP, self).__init__()
        # normalize x independently for each dimension
        self.x_mean = x.mean(0)
        self.x_std = x.std(0)
        self.x = (x - self.x_mean.expand_as(x)) / (self.x_std.expand_as(x) + EPS)

        if normal_y_mode == 0:
            # normalize y all together
            self.y_mean = y.mean()
            self.y_std = y.std()
        elif normal_y_mode == 1:
            # normalize y by each dimension
            self.y_mean = y.mean(0)
            self.y_std = y.std(0)
        elif normal_y_mode == 2:
            self.y_mean = torch.zeros(1)
            self.y_std = torch.ones(1)

        self.y = (y - self.y_mean.expand_as(y)) / (self.y_std.expand_as(y) + EPS)

        # GP hyper-parameters

        # self.log_beta = nn.Parameter(torch.ones(1) * -5)   # a large noise, ard
        self.log_beta = nn.Parameter(torch.ones(1) * -8)   # a large noise, ard
        # self.log_beta = nn.Parameter(torch.ones(1) * -6)   # a large noise, ard

        self.log_length_rbf = nn.Parameter(torch.zeros(x.shape[1]))  # RBF Kernel length
        self.log_coe_rbf = nn.Parameter(torch.zeros(1))   # RBF Kernel coefficient

        self.log_coe_linear = nn.Parameter(torch.zeros(1))  # Linear Kernel coefficient

        self.log_length_matern3 = torch.nn.Parameter(torch.zeros(x.shape[1]))  # Matern3 Kernel length
        self.log_coe_matern3 = torch.nn.Parameter(torch.zeros(1))  # Matern3 Kernel coefficient

        self.log_length_matern5 = torch.nn.Parameter(torch.zeros(x.shape[1]))  # Matern5 Kernel length
        self.log_coe_matern5 = torch.nn.Parameter(torch.zeros(1))  # Matern5 Kernel coefficient

        # debug validation
        if 'x_te' in kwargs and 'y_te' in kwargs:
            self.x_te = kwargs['x_te']
            self.y_te = kwargs['y_te']

    # customized kernel
    def kernel_customized(self, x1, x2):
        # return self.kernel_matern3(x1, x2) + self.kernel_matern5(x1, x2)
        return self.kernel_matern3(x1, x2) + self.kernel_matern5(x1, x2) + self.kernel_linear(x1, x2)
        # return self.kernel_rbf(x1, x2) + self.kernel_linear(x1, x2)

    def kernel_rbf(self, x1, x2):
        x1 = x1 / self.log_coe_rbf.exp()
        x2 = x2 / self.log_coe_rbf.exp()
        # L2 norm
        x1_norm2 = torch.sum(x1 * x1, dim=1).view(-1, 1)
        x2_norm2 = torch.sum(x2 * x2, dim=1).view(-1, 1)

        k_rbf = -2.0 * x1 @ x2.t() + x1_norm2.expand(x1.size(0), x2.size(0)) + x2_norm2.t().expand(x1.size(0), x2.size(0))
        k_rbf = self.log_coe_rbf.exp() * torch.exp(-0.5 * k_rbf)
        return k_rbf

    def kernel_linear(self, x1, x2):
        k_linear = self.log_coe_linear.exp() * (x1 @ x2.t())
        return k_linear

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

    def forward(self, x_te):
        x_te = (x_te - self.x_mean.expand_as(x_te)) / self.x_std.expand_as(x_te)

        sigma = self.kernel_customized(self.x, self.x) + self.log_beta.exp().pow(-1) * torch.eye(self.x.size(0)) \
            + JITTER * torch.eye(self.x.size(0))

        kx = self.kernel_customized(self.x, x_te)
        L = torch.cholesky(sigma)
        l_inv_kx, _ = torch.triangular_solve(kx, L, upper=False)

        # option 1
        mean = kx.t() @ torch.cholesky_solve(self.y, L)  # torch.linalg.cholesky()
        # var_diag = self.log_coe_rbf.exp().expand(n_test, 1) \
        #     - (l_inv_kx**2).sum(dim=0).view(-1, 1)
        var_diag = self.kernel_customized(x_te, x_te).diag().view(-1, 1) - (l_inv_kx**2).sum(dim=0).view(-1, 1)

        # add the noise uncertainty
        var_diag = var_diag + self.log_beta.exp().pow(-1)

        mean = mean * self.y_std.expand_as(mean) + self.y_mean.expand_as(mean)
        var_diag = var_diag.expand_as(mean) * self.y_std ** 2
        return mean, var_diag

    def negative_log_likelihood(self):
        y_num, y_dimension = self.y.shape
        sigma = self.kernel_customized(self.x, self.x) + self.log_beta.exp().pow(-1) * torch.eye(
            self.x.size(0)) + JITTER * torch.eye(self.x.size(0))

        L = torch.linalg.cholesky(sigma)
        # option 1 (use this if torch supports)
        gamma,_ = torch.triangular_solve(self.y, L, upper = False)
        # option 2
        # gamma = L.inverse() @ y       # we can use this as an alternative because L is a lower triangular matrix.

        nll = 0.5 * (gamma ** 2).sum() + L.diag().log().sum() * y_dimension \
              + 0.5 * y_num * torch.log(2 * torch.tensor(PI)) * y_dimension
        return nll

    def train_adam(self, niteration=10, lr=0.1):
        # adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        for i in range(niteration):
            optimizer.zero_grad()
            loss = self.negative_log_likelihood()
            loss.backward()
            optimizer.step()
            print('iter', i, 'nnl:{:.5f}'.format(loss.item()))

    def train_adam_debug(self, niteration=10, lr=0.1, fig_pth='./MSE.png'):
        # adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        mse_list = []
        for i in range(niteration):
            optimizer.zero_grad()
            loss = self.negative_log_likelihood()
            loss.backward()
            optimizer.step()
            print('iter', i, 'nnl:{:.5f}'.format(loss.item()))
            mse_list.append(self.valid())
        self.plot_validation(niteration, mse_list, fig_pth)

    def train_bfgs(self, niteration=50, lr=0.1):
        # LBFGS optimizer
        optimizer = torch.optim.LBFGS(self.parameters(), lr=lr)
        for i in range(niteration):
            def closure():
                optimizer.zero_grad()
                loss = self.negative_log_likelihood()
                loss.backward()
                print('iter', i, 'nnl:{:.5f}'.format(loss.item()))
                return loss

            optimizer.step(closure)

    def valid(self) -> float:
        # for debug, abort this when published
        y_mean, _ = self(self.x_te)
        mse = mean_squared_error(self.y_te.detach(), y_mean.detach())
        return mse

    @ staticmethod
    def plot_validation(iter_n: int, mse_list: list, fig_pth: str):
        # for debug, abort this when published
        plt.plot(list(range(iter_n)), mse_list, label='MSE', color='navy')
        plt.legend()
        plt.grid()
        plt.gcf().savefig(fig_pth)
        plt.show()
        plt.close('all')

    # TODO: add conjugate gradient method


if __name__ == "__main__":
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

    print(xte.shape)
    print(yte.shape)
    print(xtr.shape)
    print(ytr.shape)

    model = CIGP(xtr, ytr, 1)
    model.train_adam(200, lr=0.1)
    # model.train_bfgs(50, lr=0.001)

    with torch.no_grad():
        ypred, ystd = model(xte)

    # plt.errorbar(xte, ypred.detach(), ystd.sqrt().squeeze().detach(),fmt='r-.' ,alpha = 0.2)
    plt.plot(xte, ypred.detach(),'r-.', label='ypred')
    plt.plot(xtr, ytr, 'b+', label='ytrain')
    plt.plot(xte, yte, 'k-', label='ytest')
    plt.legend()
    plt.show()

    # plt.close('all')
    plt.plot(xtr, ytr, 'b+')
    for i in range(3):
        plt.plot(xte, yte[:, i], label='truth', color='r')
        plt.plot(xte, ypred[:, i], label='prediction', color='navy')
        plt.fill_between(xte.squeeze(-1).detach().numpy(),
                         ypred[:, i].squeeze(-1).detach().numpy() + torch.sqrt(ystd[:, i].squeeze(-1)).detach().numpy(),
                         ypred[:, i].squeeze(-1).detach().numpy() - torch.sqrt(ystd[:, i].squeeze(-1)).detach().numpy(),
                         alpha=0.2)
    plt.show()

# %%
