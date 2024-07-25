
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from model_FAQ.non_positive_definite_fixer import remove_similar_data
import core.GP_CommonCalculation as GP
from data_sample import generate_example_data as data
from core.kernel import NeuralKernel,ARDKernel
import time
from core.cigp_v10 import cigp
print(torch.__version__)
# I use torch (1.11.0) for this work. lower version may not work.

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Fixing strange error if run in MacOS
JITTER = 1e-3
EPS = 1e-10
PI = 3.1415
torch.set_default_dtype(torch.float64)
class autoGP(nn.Module):

    def __init__(self, X, Y,kernel=None, normal_method='min_max', inputwarp=False,num_inducing=None,deepkernel=False):
        super(autoGP, self).__init__()

        self.data = GP.DataNormalization(method=normal_method)
        self.data.fit(X, 'x')
        self.data.fit(Y, 'y')
        # self.X = self.data.normalize(X, 'x')
        self.Y = self.data.normalize(Y, 'y')

        self.X=X

        # GP hyperparameters
        self.log_beta = nn.Parameter(torch.ones(1) * -4)  # Initial noise level
        self.inputwarp=inputwarp
        # Inducing points
        input_dim = self.X.size(1)

        if kernel is None:
            self.kernel = NeuralKernel(input_dim)
        else:
            self.kernel=kernel

        if input_dim>2:
            self.deepkernel=True
            self.kernel=ARDKernel(input_dim)

        else:
            self.deepkernel=deepkernel





        if self.deepkernel:
            self.FeatureExtractor = torch.nn.Sequential(nn.Linear(input_dim, input_dim * 10),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(input_dim * 10, input_dim * 5),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(input_dim * 5, input_dim))
            self.inputwarp=False  # if deepkernel is enabled then inputwarp is disabled
        else:
            self.FeatureExtractor = lambda x: x # Do nothing
            self.inputwarp=True # if deepkernel is disabled then inputwarp is enabled

        if self.inputwarp:
            self.warp = GP.Warp(method='kumar', initial_a=1.0, initial_b=1.0)
        else:
            self.warp = GP.Warp(method='unchange', initial_a=1.0, initial_b=1.0)

        if num_inducing is None:
            num_inducing=self.X.size(0)*input_dim//20



        self.xm = nn.Parameter(torch.rand((num_inducing, input_dim)))  # Inducing points

    def negative_lower_bound(self):
        """Negative lower bound as the loss function to minimize."""
        #X=self.warp.transform(self.X)
        if self.deepkernel:
            X1=self.FeatureExtractor(self.X)
            xm1=self.FeatureExtractor(self.xm)

            X=(X1-X1.mean(0).expand_as(X1))/X1.std(0).expand_as(X1)
            xm=(xm1-xm1.mean(0).expand_as(xm1))/xm1.std(0).expand_as(xm1)

        elif self.inputwarp:
            X=self.warp.transform(self.X)
            xm=self.xm

        else:
            X=self.X
            xm=self.xm

        n = self.X.size(0)
        K_mm = self.kernel(xm, xm) + JITTER * torch.eye(self.xm.size(0))
        L = torch.linalg.cholesky(K_mm)
        K_mn = self.kernel(xm, X)
        K_nn = self.kernel(X, X)
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
        #X = self.warp.transform(self.X)
        if self.deepkernel:
            X1=self.FeatureExtractor(self.X)
            X = (X1 - X1.mean(0).expand_as(X1)) / X1.std(0).expand_as(X1)
            xm1 = self.FeatureExtractor(self.xm)
            xm=(xm1-xm1.mean(0).expand_as(xm1))/xm1.std(0).expand_as(xm1)
        else:
            xm=self.xm
            X=self.X

        K_mm = self.kernel(xm, xm) + JITTER * torch.eye(self.xm.size(0))
        L = torch.linalg.cholesky(K_mm)
        L_inv = torch.inverse(L)
        K_mm_inv = L_inv.t() @ L_inv

        K_mn = self.kernel(xm, X)
        K_nm = K_mn.t()
        sigma = torch.inverse(K_mm + self.log_beta.exp() * K_mn @ K_nm)

        mean_m = self.log_beta.exp() * (K_mm @ sigma @ K_mn) @ self.Y
        A_m = K_mm @ sigma @ K_mm
        return mean_m, A_m, K_mm_inv

    def forward(self, Xte):
        """Compute mean and variance for posterior distribution."""
        self.data.fit(Xte, 'xte')
        Xte = self.data.normalize(Xte, 'x')  # we have to make sure xte is in [0,1]
        if self.deepkernel:
            X1 = self.FeatureExtractor(self.X)
            #Xte = self.warp.transform(Xte)
            Xte1=self.FeatureExtractor(Xte)
            Xte = (Xte1 - X1.mean(0).expand_as(Xte)) / X1.std(0).expand_as(Xte)

            xm1 = self.FeatureExtractor(self.xm)
            xm=(xm1-xm1.mean(0).expand_as(xm1))/xm1.std(0).expand_as(xm1)
        else:
            xm = self.xm
        K_tt = self.kernel(Xte, Xte)
        K_tm = self.kernel(Xte, xm)
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

    def train_auto(self,niteration1=10,lr1=0.01,niteration2=100,lr2=0.001):
        start_time = time.time()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr1)
        optimizer.zero_grad()
        for i in range(niteration1):
            optimizer.zero_grad()
            loss = self.negative_lower_bound()
            loss.backward(retain_graph=True)
            optimizer.step()
            #if i%10==0:
               #print(self.warp.transform(self.Y)[:2])
                #print(self.warp.a,self.warp.b)
            print('train with adam, iter', i, ' nll:', loss.item())

        optimizer = torch.optim.LBFGS(self.parameters(), max_iter=niteration2, lr=lr2)
        def closure():
            optimizer.zero_grad()
            loss = self.negative_lower_bound()
            loss.backward(retain_graph=True)  # Retain the graph
            print('train with LBFGS, nll:', loss.item())

            return loss

        #optimizer.step(closure)
        end_time = time.time()  # 结束时间
        training_time = end_time - start_time
        print(f'AutoGP training completed in {training_time:.2f} seconds')
if __name__ == "__main__":
    # np.random.seed(0)
    # torch.manual_seed(0)
    # n = 500
    # X = np.random.rand(n, 1) * 10
    # Y = np.sin(X) + np.random.randn(n, 1) * 0.1
    # X = torch.tensor(X, dtype=torch.float64)
    # Y = torch.tensor(Y, dtype=torch.float64)
    # Xte = torch.linspace(0, 10, 1000).view(-1, 1)
    # Yte = torch.sin(Xte)
    xtr, ytr, xte, yte = data.generate(1000, 1000, seed=2, input_dim=1)
    model = autoGP(xtr, ytr, deepkernel=False)
    model.train_auto()

    mean, var = model.forward(xte)
    mse= torch.mean((mean - yte) ** 2)
    print(mse)
    for name, param in model.named_parameters():
         print(name, param)
    print(model.deepkernel)
    # plt.plot(Xte, mean.detach().numpy(), 'r')
    # plt.plot(Xte, Yte, 'g')
    # plt.errorbar(Xte.numpy().reshape(300), mean.detach().numpy().reshape(300),
    #              yerr=var.sqrt().squeeze().detach().numpy(), fmt='r-.', alpha=0.2)
    # plt.scatter(X.numpy(), Y.numpy(),color='b',s=10)
    # plt.show()
