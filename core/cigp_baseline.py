import torch
import torch.nn as nn
from core.kernel import ARDKernel,NeuralKernel,PeriodicKernel
import numpy as np
import matplotlib.pyplot as plt
from data_sample import generate_example_data as data
import core.GP_CommonCalculation as GP
JITTER= 1e-6
EPS= 1e-10
PI= 3.1415
torch.set_default_dtype(torch.float64)
class cigp(nn.Module):
    def __init__(self, X, Y,kernel=ARDKernel,normal_y_mode=0,K_inv_method='cholesky'):
        super(cigp, self).__init__()

        #normalize X independently for each dimension
        self.Xmean = X.mean(0)
        self.Xstd = X.std(0)
        self.X = (X - self.Xmean.expand_as(X)) / (self.Xstd.expand_as(X) + EPS)
        self.kernel=kernel(input_dim=X.size(1))
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
        self.log_beta = nn.Parameter(torch.ones(1) * -4)   # a large noise by default. Smaller value makes larger noise variance.
        self.K_inv_method=K_inv_method
        self.device=self.X.device
    def forward(self, Xte):
        n_test = Xte.size(0)
        Xte = (Xte - self.Xmean.expand_as(Xte)) / self.Xstd.expand_as(Xte)

        Sigma = self.kernel(self.X, self.X) + self.log_beta.exp().pow(-1) * torch.eye(self.X.size(0), device=self.device) \
                + JITTER * torch.eye(self.X.size(0),device=self.device)

        k_xt = self.kernel(self.X, Xte)
        k_tt = self.kernel(Xte, Xte)

        mean,var=GP.conditional_Gaussian(self.Y,Sigma,k_xt,k_tt,Kinv_method=self.K_inv_method)
        var_diag=var.diag().view(-1,1)+ self.log_beta.exp().pow(-1)
        # De-normalized
        mean = mean * self.Ystd.expand_as(mean) + self.Ymean.expand_as(mean)
        var_diag = var_diag.expand_as(mean) * self.Ystd ** 2
        return mean,var_diag

    def negative_log_likelihood(self):
        Sigma = self.kernel(self.X, self.X) + self.log_beta.exp().pow(-1) * torch.eye(
            self.X.size(0),device=self.device) + JITTER * torch.eye(self.X.size(0),device=self.device)

        return -GP.Gaussian_log_likelihood(self.Y, Sigma, Kinv_method=self.K_inv_method)

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
            if i % 10 == 0:
                print('iter', i, 'nll:{:.5f}'.format(loss.item()))


# %%
if __name__ == "__main__":
    print('testing')
    print(torch.__version__)
    torch.manual_seed(seed=2)
    # single output test 1
    xte = torch.linspace(0, 6, 100).view(-1, 1)
    yte = torch.sin(xte) + 10

    xtr = torch.rand(16, 1) * 6
    ytr = torch.sin(xtr) + torch.randn(16, 1) * 0.5 + 10

    model = cigp(xtr, ytr)
    model.train_adam(200, lr=0.1)
    # model.train_bfgs(50, lr=0.1)

    with torch.no_grad():
        ypred, ypred_var = model(xte)

    plt.errorbar(xte, ypred.reshape(-1).detach(), ypred_var.sqrt().squeeze().detach(), fmt='r-.', alpha=0.2)
    plt.plot(xtr, ytr, 'b+')
    plt.show()

    # single output test 2
    xte = torch.rand(128, 2) * 2
    yte = torch.sin(xte.sum(1)).view(-1, 1) + 10

    xtr = torch.rand(32, 2) * 2
    ytr = torch.sin(xtr.sum(1)).view(-1, 1) + torch.randn(32, 1) * 0.5 + 10

    model = cigp(xtr, ytr)
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
                        xte.tanh()])

    xtr = torch.rand(32, 1) * 6
    ytr = torch.sin(xtr) + torch.rand(32, 1) * 0.5
    ytr = torch.hstack([torch.sin(xtr),
                        torch.cos(xtr),
                        xtr.tanh()]) + torch.randn(32, 3) * 0.2

    model = cigp(xtr, ytr, normal_y_mode=1)
    model.train_adam(100, lr=0.1)
    # model.train_bfgs(50, lr=0.001)

    with torch.no_grad():
        ypred, ypred_var = model(xte)

    # plt.errorbar(xte, ypred.detach(), ypred_var.sqrt().squeeze().detach(),fmt='r-.' ,alpha = 0.2)
    plt.plot(xte.numpy(), ypred.detach().numpy(), 'r-.')
    plt.plot(xtr.numpy(), ytr.numpy(), 'b+')
    plt.plot(xte.numpy(), yte.numpy(), 'k-')
    plt.show()

    # plt.close('all')
    plt.plot(xtr.numpy(), ytr.numpy(), 'b+')
    for i in range(3):
        plt.plot(xte.numpy(), yte[:, i].numpy(), label='truth', color='r')
        plt.plot(xte.numpy(), ypred[:, i].detach().numpy(), label='prediction', color='navy')
        plt.fill_between(xte.squeeze(-1).detach().numpy(),
                         ypred[:, i].squeeze(-1).detach().numpy() + torch.sqrt(
                             ypred_var[:, i].squeeze(-1)).detach().numpy(),
                         ypred[:, i].squeeze(-1).detach().numpy() - torch.sqrt(
                             ypred_var[:, i].squeeze(-1)).detach().numpy(),
                         alpha=0.2)
    plt.show()

# %%

