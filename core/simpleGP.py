# # Conditional independent Gaussian process (CIGP) for vector output regression based on pytorch
# #
# # CIGP use a single kernel for each output. Thus the log likelihood is simply a sum of the log likelihood of each output.
import torch
import torch.nn as nn
from core.kernel import ARDKernel
import matplotlib.pyplot as plt
import core.GP_CommonCalculation as GP

JITTER = 1e-6
EPS = 1e-10
PI = 3.1415
torch.set_default_dtype(torch.float64)


class cigp(nn.Module):
    def __init__(self, kernel=ARDKernel(1), log_beta=None, K_inv_method='cholesky'):
        super(cigp, self).__init__()
        # GP hyperparameters
        if log_beta is None:
            self.log_beta = nn.Parameter(torch.ones(1) * -4)
        else:
            self.log_beta = nn.Parameter(log_beta)
        self.kernel = kernel
        self.K_inv_method = K_inv_method

    def forward(self, xtr, ytr, xte):
        Sigma = self.kernel(xtr, xtr) + self.log_beta.exp().pow(-1) * torch.eye(xtr.size(0),
                                                                                device=self.log_beta.device) \
                + JITTER * torch.eye(xtr.size(0), device=self.log_beta.device)

        k_xt = self.kernel(xtr, xte)
        k_tt = self.kernel(xte, xte)

        mean, var = GP.conditional_Gaussian(ytr, Sigma, k_xt, k_tt, Kinv_method=self.K_inv_method)
        var_diag = var.diag().view(-1, 1) + self.log_beta.exp().pow(-1)

        return mean, var_diag

    def negative_log_likelihood(self, xtr, ytr):
        Sigma = self.kernel(xtr, xtr) + self.log_beta.exp().pow(-1) * torch.eye(
            xtr.size(0), device=self.log_beta.device) + JITTER * torch.eye(xtr.size(0), device=self.log_beta.device)

        return -GP.Gaussian_log_likelihood(ytr, Sigma, Kinv_method=self.K_inv_method)

    def train_adam(self, xtr, ytr, niteration=10, lr=0.1):
        # adam optimizer
        # uncommont the following to enable
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        for i in range(niteration):
            optimizer.zero_grad()
            # self.update()
            loss = self.negative_log_likelihood(xtr, ytr)
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

    normalizer = GP.DataNormalization()
    normalizer.fit(xtr, 'x')
    normalizer.fit(ytr, 'y')
    xtr_normalized = normalizer.normalize(xtr, 'x')
    ytr_normalized = normalizer.normalize(ytr, 'y')
    xte_normalized = normalizer.normalize(xte, 'x')
    model = cigp()
    model.train_adam(xtr_normalized, ytr_normalized, 200, lr=0.1)

    with torch.no_grad():
        ypred, ypred_var = model.forward(xtr_normalized, ytr_normalized, xte_normalized)
        ypred = normalizer.denormalize(ypred, 'y')
        ypred_var = normalizer.denormalize_cov(ypred_var, 'y')
    plt.errorbar(xte, ypred.reshape(-1).detach(), ypred_var.sqrt().squeeze().detach(), fmt='r-.', alpha=0.2)
    plt.plot(xtr, ytr, 'b+')
    plt.show()

    # single output test 2
    xte = torch.rand(128, 2) * 2
    yte = torch.sin(xte.sum(1)).view(-1, 1) + 10

    xtr = torch.rand(32, 2) * 2
    ytr = torch.sin(xtr.sum(1)).view(-1, 1) + torch.randn(32, 1) * 0.5 + 10

    normalizer = GP.DataNormalization()
    normalizer.fit(xtr, 'x')
    normalizer.fit(ytr, 'y')
    xtr_normalized = normalizer.normalize(xtr, 'x')
    ytr_normalized = normalizer.normalize(ytr, 'y')
    xte_normalized = normalizer.normalize(xte, 'x')
    model = cigp()
    model.train_adam(xtr_normalized, ytr_normalized, 200, lr=0.1)

    with torch.no_grad():
        ypred, ypred_var = model.forward(xtr_normalized, ytr_normalized, xte_normalized)
        ypred = normalizer.denormalize(ypred, 'y')
        ypred_var = normalizer.denormalize_cov(ypred_var, 'y')

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

    normalizer = GP.DataNormalization(mode=0)
    normalizer.fit(xtr, 'x')
    normalizer.fit(ytr, 'y')
    xtr_normalized = normalizer.normalize(xtr, 'x')
    ytr_normalized = normalizer.normalize(ytr, 'y')
    xte_normalized = normalizer.normalize(xte, 'x')
    model = cigp()
    model.train_adam(xtr_normalized, ytr_normalized, 200, lr=0.1)

    with torch.no_grad():
        ypred, ypred_var = model.forward(xtr_normalized, ytr_normalized, xte_normalized)
        ypred = normalizer.denormalize(ypred, 'y')
        print(ypred_var.expand_as(ypred).size())
        ypred_var = normalizer.denormalize_cov(ypred_var.expand_as(ypred), 'y')

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
