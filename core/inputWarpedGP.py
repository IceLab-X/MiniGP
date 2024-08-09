import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import core.GP_CommonCalculation as GP
print(torch.__version__)
# I use torch (1.11.0) for this work. lower version may not work.
from core.kernel import ARDKernel
import os
from core.cigp_baseline import cigp
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Fixing strange error if run in MacOS
JITTER = 1e-6
EPS = 1e-10
PI = 3.1415

class inputwarp_gp(nn.Module):
    def __init__(self, X, Y, normal_y_mode=0):
        super(inputwarp_gp, self).__init__()

        #normalize X independently for each dimension
        self.X=X
        self.Y=Y
        self.normalizer=GP.DataNormalization(method='min_max')
        self.normalizer.fit(self.X,'x')
        self.normalizer.fit(self.Y,'y')
        self.X=self.normalizer.normalize(self.X,'x')
        self.Y=self.normalizer.normalize(self.Y,'y')

        # GP hyperparameters
        self.log_beta = nn.Parameter(torch.ones(1) *-4)   # a large noise by default. Smaller value makes larger noise variance.
        self.kernel=ARDKernel(1)   # ARD length scale
        self.warp=GP.Warp(method='kumar', initial_a=1.0, initial_b=1.0)

    def forward(self, Xte):
        X=self.warp.transform(self.X)

        self.normalizer.fit(Xte,'xte')
        Xte = self.normalizer.normalize(Xte, 'xte')
        Xte=self.warp.transform(Xte)
        Sigma = self.kernel(X, X) + self.log_beta.exp().pow(-1) * torch.eye(self.X.size(0)) \
            + JITTER * torch.eye(self.X.size(0))

        kx = self.kernel(X, Xte)
        L = torch.linalg.cholesky(Sigma)
        LinvKx = torch.linalg.solve_triangular(L, kx, upper=False)

        # Option 1
        mean = kx.t() @ torch.cholesky_solve(self.Y, L)

        var_diag = self.kernel(Xte, Xte).diag().view(-1, 1) \
            - (LinvKx**2).sum(dim=0).view(-1, 1)

        # Add the noise uncertainty
        var_diag = var_diag + self.log_beta.exp().pow(-1)

        # De-normalized
        mean=self.normalizer.denormalize(mean,'y')
        var_diag=self.normalizer.denormalize_cov(var_diag,'y')

        return mean, var_diag

    def negative_log_likelihood(self):
        X=self.warp.transform(self.X)
        y_num, y_dimension = self.Y.shape
        Sigma = self.kernel(X, X) + self.log_beta.exp().pow(-1) * torch.eye(
            X.size(0)) + JITTER * torch.eye(X.size(0))

        L = torch.linalg.cholesky(Sigma)

        Gamma = torch.linalg.solve_triangular(L,self.Y, upper=False)

        nll =  0.5 * (Gamma ** 2).sum() +  L.diag().log().sum() * y_dimension  \
            + 0.5 * y_num * torch.log(2 * torch.tensor(PI)) * y_dimension
        return nll

    def train_adam(self, niteration=10, lr=0.1):
        #adam optimizer

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        for i in range(niteration):
            optimizer.zero_grad()
            # self.update()
            loss = self.negative_log_likelihood()
            loss.backward(retain_graph=True)
            optimizer.step()
            # print('loss_nll:', loss.item())
            # print('iter', i, ' nll:', loss.item())
            if i % 10 == 0:
                print('iter', i, 'nll:{:.5f}'.format(loss.item()))
        # optimizer = torch.optim.Adam(self.warp.parameters(), lr=lr/10)
        # optimizer.zero_grad()
        # for i in range(niteration):
        #     optimizer.zero_grad()
        #     # self.update()
        #     loss = self.negative_log_likelihood()
        #     loss.backward(retain_graph=True)
        #     optimizer.step()
        #     # print('loss_nll:', loss.item())
        #     # print('iter', i, ' nll:', loss.item())
        #     if i % 10 == 0:
        #         print('iter', i, 'nll:{:.5f}'.format(loss.item()))
        # optimizer = torch.optim.Adam(list(self.kernel.parameters()) + [self.log_beta], lr=lr)
        # optimizer.zero_grad()
        # for i in range(niteration):
        #     optimizer.zero_grad()
        #     # self.update()
        #     loss = self.negative_log_likelihood()
        #     loss.backward(retain_graph=True)
        #     optimizer.step()
        #     # print('loss_nll:', loss.item())
        #     # print('iter', i, ' nll:', loss.item())
        #     if i % 10 == 0:
        #         print('iter', i, 'nll:{:.5f}'.format(loss.item()))
if __name__ == '__main__':
    # Example usage
    # Generate some data
    np.random.seed(0)
    torch.manual_seed(0)
    n = 100
    X = np.random.rand(n, 1) *10
    Y = np.sin(X) + np.random.randn(n, 1) * 0.1
    X = torch.tensor(X, dtype=torch.float64)
    Y = torch.tensor(Y, dtype=torch.float64)
    # Create the model
    model = inputwarp_gp(X, Y)
    model2= cigp(X,Y)
    # Train the model
    model.train_adam(niteration=100, lr=0.1)
    model.train_adam(niteration=100, lr=0.1)
    # Test the model
    Xte = torch.linspace(0, 10, 1000).view(-1, 1)
    Yte=torch.sin(Xte)

    mean, var = model(Xte)
    mean2,var2=model2.forward(Xte)
    # Plot the results
    warp=model.warp
    plt.figure()
    normalized_X= model.normalizer.normalize(X,'x')
    plt.scatter(X,Y)
    plt.plot(Xte.numpy(), mean.detach().numpy(), 'r')
    print(model.warp.a,model.warp.b)


    plt.show()

    mse = torch.mean((mean - Yte) ** 2)
    print(f'Test warped MSE: {mse.item()}')
    mse2 = torch.mean((mean2 - Yte) ** 2)
    print(f'Test MSE: {mse2.item()}')
    R_square=1-torch.sum((mean-Yte)**2)/torch.sum((Yte-torch.mean(Yte))**2)
    print('R square:',R_square.item())
    R_square2=1-torch.sum((mean2-Yte)**2)/torch.sum((Yte-torch.mean(Yte))**2)
    print('R square2:',R_square2.item())