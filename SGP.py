import torch
import torch.nn as nn
import numpy as np

from matplotlib import pyplot as plt
import os
import kernel
from torch.utils.data import TensorDataset, DataLoader



# Variational Sparse Gaussian Processes

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Fixing strange error if run in MacOS

JITTER = 1e-2

EPS = 1e-10
PI = 3.1415

class VSGP(nn.Module):
    def __init__(self, X, Y, dim, num_inducing):
        super(VSGP, self).__init__()
        self.kernel1 = kernel.ARDKernel(dim)
        self.X = X
        self.Y = Y

        # GP hyperparameters
        self.log_beta = nn.Parameter(torch.ones(1) * -4)  # a large noise by default. Smaller value makes larger noise variance.

        self.log_scale = nn.Parameter(torch.zeros(1))  # kernel scale
        # inducing point
        self.xm=nn.Parameter(torch.rand(num_inducing, dim)) # inducing point


    def kernel(self, X1, X2):
        # define kernel function
        out = self.kernel1(X1, X2)
        return out

    def negative_log_likelihood(self): # negtive lower bound as the loss function we want to minimize later

        n = self.X.size(0)

        K_mm = self.kernel(self.xm, self.xm) + JITTER * torch.eye(self.xm.size(0))
        K_mn = self.kernel(self.xm, self.X)
        K_nn = self.kernel(self.X, self.X)
        L = torch.linalg.cholesky(K_mm)
        A, _ = torch.triangular_solve(K_mn, L, upper=False)
        A = A * torch.sqrt(self.log_beta.exp())
        AAT = A @ A.t()
        #print(AAT)
        B = torch.eye(self.xm.size(0)) + AAT
        B+= JITTER * torch.eye(self.xm.size(0))*B.mean()
        LB = torch.linalg.cholesky(B)

        c, _ = torch.triangular_solve(A @ self.Y, LB, upper=False)
        c = c * torch.sqrt(self.log_beta.exp())
        nll = n / 2 * torch.log(2 * torch.tensor(PI)) + torch.sum(torch.log(torch.diagonal(LB))) + \
              n / 2 * torch.log(1 / self.log_beta.exp()) + self.log_beta.exp() / 2 * torch.sum(self.Y * self.Y) - \
              0.5 * torch.sum(c.squeeze() * c.squeeze()) + self.log_beta.exp() / 2 * torch.sum(torch.diagonal(K_nn))\
              - 0.5 * torch.trace(AAT)
        return nll

    def train_adam(self, niteration=10, lr=0.1):

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
            # print('iter', i, 'nll:{:.5f}'.format(loss.item()))


    def forward(self, xte):


        # compute mean for posterior distribution
        K_mm = self.kernel(self.xm, self.xm) + JITTER * torch.eye(self.xm.size(0))
        K_mm_inv = torch.inverse(K_mm)
        K_mn = self.kernel(self.xm, self.X)
        K_nm = K_mn.t()
        sigma = torch.inverse(K_mm + self.log_beta.exp() * K_mn @ K_nm)

        mean_m = self.log_beta.exp() * (K_mm @ sigma @ K_mn) @ self.Y
        A_m = K_mm @ sigma @ K_mm

        K_tt = self.kernel(xte, xte)
        K_tm = self.kernel(xte, self.xm)
        K_mt = K_tm.t()
        mean = (K_tm @ K_mm_inv) @ mean_m
        var= K_tt - K_tm @ K_mm_inv @ K_mt + K_tm @ K_mm_inv @ A_m @ K_mm_inv @ K_mt
        var_diag = var.diag().view(-1, 1)
        return mean, var_diag
class SVGP(nn.Module):
    def __init__(self, X, Y, batch_size,dim, num_inducing):
        super(SVGP, self).__init__()
        self.kernel1 = kernel.ARDKernel(dim)
        self.X = X
        self.Y = Y
        # Restructure the data for mini-batch
        self.dataset = TensorDataset(self.X, self.Y)
        self.batch_size = batch_size
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # inducing point as subset_set_of_training_set
        self.num_inducing = num_inducing
        subset_indices = torch.randperm(len(self.X))[:self.num_inducing]
        xm = self.X[subset_indices]
        ym = self.Y[subset_indices]
        # GP hyperparameters
        self.log_beta = nn.Parameter(torch.ones(1) * -4)  # a large noise by default. Smaller value makes larger noise variance.

        self.log_scale = nn.Parameter(torch.zeros(1))  # kernel scale
        # inducing point
        self.xm=nn.Parameter(torch.rand(num_inducing, dim)) # inducing point
        mean = ym.clone()
        self.q_u_mean = nn.Parameter(mean)
        self.chole = nn.Parameter(torch.rand(xm.size(0)).unsqueeze(1))
    def kernel(self, X1, X2):
        # define kernel function
        out = self.kernel1(X1, X2)
        return out


    def negative_lower_bound(self,xtr_batch,ytr_batch):


        k_mm = self.kernel(self.xm, self.xm) + JITTER * torch.eye(self.xm.size(0))
        k_mn = self.kernel(self.xm, xtr_batch)
        Lm = torch.linalg.cholesky(k_mm)
        k_nm = k_mn.t()

        gamma, _ = torch.triangular_solve(k_mn, Lm, upper=False)
        K = gamma.t() @ gamma
        K_tilda = (self.kernel(xtr_batch, self.X).diag() - K.diag()).view(-1, 1)

        # Option 1
        # k_mm_inv=torch.inverse(k_mm)

        # Option 2 More efficient and stable
        self.k_mm_inv = torch.cholesky_inverse(Lm)

        self.q_u_S = self.chole @ self.chole.t() + JITTER * torch.eye(self.xm.size(0))
        Ls = torch.linalg.cholesky(self.q_u_S)



        k_i = k_nm.unsqueeze(2)  # Reshape k_nm to accommodate the batch dimension [b,m,1]

        A_i = self.log_beta.exp() * (self.k_mm_inv @ k_i @ k_i.transpose(1, 2) @ self.k_mm_inv)
        SA_i = self.q_u_S @ A_i
        tr_SA_i = torch.einsum('bii->b', SA_i) # trace of SA_i
        L = -0.5 * self.batch_size * torch.log(2 * PI / self.log_beta.exp()) \
            - 0.5 * self.log_beta.exp() * ((ytr_batch - k_i.transpose(1, 2) @ self.k_mm_inv @ self.q_u_mean) ** 2).sum(dim=0).view(-1, 1) \
            - 0.5 * self.log_beta.exp() * K_tilda.sum(dim=0).view(-1, 1) - 0.5 * (tr_SA_i.sum(dim=0).view(-1, 1))

        # compute KL
        logdetS = 2 * Ls.diag().abs().log().sum()
        logdetKmm = 2 * Lm.diag().log().sum()
        k_mm_invS = self.k_mm_inv @ self.q_u_S

        KL = 0.5 * k_mm_invS.diag().sum(dim=0).view(-1, 1) + 0.5 * (
                    self.q_u_mean.t() @ self.k_mm_inv @ self.q_u_mean) - 0.5 * logdetS + 0.5 * logdetKmm - 0.5 * self.num_inducing

        return KL, L, self.k_mm_inv



    def train_adam(self, niteration=10, lr=0.1):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)


        n = self.X.size(0)
        for epoch in range(niteration):
            epoch_loss = 0  # To accumulate loss over each epoch

            for xtr_batch, ytr_batch in self.data_loader:
                optimizer.zero_grad()  # Zero the gradients to prevent accumulation

                # Compute the batch loss and the inverse kernel matrix (k_mm_inv, which will be used for prediction)
                KL, L, self.k_mm_inv = self.negative_lower_bound(xtr_batch, ytr_batch)
                batch_loss = (KL * self.batch_size / self.X.size(0) - L).sum(dim=0).view(-1, 1)

                batch_loss.backward()  # Backpropagate the loss to compute gradients
                optimizer.step()  # Update the model parameters

                epoch_loss += batch_loss.item()  # Sum up the loss over the epoch

            # Print the average loss for the epoch
        #print(f'Epoch {epoch}, Average Loss: {epoch_loss / len(data_loader)}')

    def forward(self, xte):
        K_tt = self.kernel(xte, xte)
        K_tm = self.kernel(xte, self.xm)
        K_mt = K_tm.t()
        ypred = (K_tm @ self.k_mm_inv) @ self.q_u_mean
        yvar = K_tt-K_tm@self.k_mm_inv@K_mt+K_tm@self.k_mm_inv@self.q_u_S@self.k_mm_inv@K_mt
        yvar = yvar.diag().view(-1,1)
        return ypred ,yvar

