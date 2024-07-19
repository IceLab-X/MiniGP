import torch
import torch.nn as nn
from core.kernel import ARDKernel
JITTER= 1e-6
EPS= 1e-10
PI= 3.1415
torch.set_default_dtype(torch.float64)
class cigp(nn.Module):
    def __init__(self, X, Y,kernel, normal_y_mode=0):
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
        self.log_beta = nn.Parameter(torch.ones(1) * -4)   # a large noise by default. Smaller value makes larger noise variance.
        self.kernel=kernel

    def forward(self, Xte):
        n_test = Xte.size(0)
        Xte = (Xte - self.Xmean.expand_as(Xte)) / self.Xstd.expand_as(Xte)

        Sigma = self.kernel(self.X, self.X) + self.log_beta.exp().pow(-1) * torch.eye(self.X.size(0)) \
            + JITTER * torch.eye(self.X.size(0))

        kx = self.kernel(self.X, Xte)
        L = torch.linalg.cholesky(Sigma)
        LinvKx = torch.linalg.solve_triangular(L, kx, upper=False)

        # Option 1
        mean = kx.t() @ torch.cholesky_solve(self.Y, L)

        var_diag = self.kernel(Xte, Xte).diag().view(-1, 1) \
            - (LinvKx**2).sum(dim=0).view(-1, 1)

        # Add the noise uncertainty
        var_diag = var_diag + self.log_beta.exp().pow(-1)

        # De-normalized
        mean = mean * self.Ystd.expand_as(mean) + self.Ymean.expand_as(mean)
        var_diag = var_diag.expand_as(mean) * self.Ystd**2

        return mean, var_diag

    def negative_log_likelihood(self):
        y_num, y_dimension = self.Y.shape
        Sigma = self.kernel(self.X, self.X) + self.log_beta.exp().pow(-1) * torch.eye(
            self.X.size(0)) + JITTER * torch.eye(self.X.size(0))

        L = torch.linalg.cholesky(Sigma)

        Gamma = torch.linalg.solve_triangular(L,self.Y, upper=False)

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
            if i % 10 == 0:
                print('iter', i, 'nll:{:.5f}'.format(loss.item()))
if __name__ == '__main__':
    # test cigp
    X = torch.rand(10, 1)
    Y = torch.sin(X) + 0.1 * torch.randn(10, 1)
    kernel = ARDKernel(1)
    model = cigp(X, Y, kernel)
    model.train_adam(100)
    Xte = torch.linspace(0, 1, 100).view(-1, 1)
    mean, var = model(Xte)
    print(mean)
    print(var)
    print('done')
