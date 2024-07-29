# Author: Zidong Chen
# Date: 2024/07/17
# This is the implementation of the Stochastic Variational Gaussian Process (SVGP) model. Key references: GP for big data

import os
import torch
import torch.nn as nn
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Fixing strange error if run in MacOS
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from core.kernel import ARDKernel
import core.GP_CommonCalculation as GP
import data_sample.generate_example_data as data
JITTER = 1e-3
PI = 3.1415
torch.manual_seed(4)


class SVIGP(nn.Module):
    def __init__(self, X, Y, num_inducing, batchsize=None):
        super(SVIGP, self).__init__()

        self.X_all, self.Y_all = X,Y
        self.batchsize = batchsize
        self.kernel = ARDKernel(1)
        self.num_inducing = num_inducing
        input_dim=X.size(1)

        # Inducing points
        self.xm = nn.Parameter(torch.rand(num_inducing, input_dim, dtype=torch.float64))  # Inducing points
        self.qu_mean = nn.Parameter(torch.zeros(num_inducing, 1, dtype=torch.float64))
        self.chole = nn.Parameter(torch.rand(self.num_inducing, 1, dtype=torch.float64))

        # kernel
        self.kernel = ARDKernel(input_dim)
        # Gaussian noise
        self.log_beta = nn.Parameter(torch.ones(1, dtype=torch.float64) * 0)

        # normalize
        self.normalizer = GP.DataNormalization(method='standard')
        self.normalizer.fit(self.X_all,'x')
        self.normalizer.fit(self.Y_all,'y')
        self.X_all = self.normalizer.normalize(self.X_all,'x')
        self.Y_all = self.normalizer.normalize(self.Y_all,'y')

        if self.batchsize is not None:
            # Create TensorDataset and DataLoader for minibatch training
            dataset = TensorDataset(self.X_all, self.Y_all)
            self.dataloader = DataLoader(dataset, batch_size=self.batchsize, shuffle=True)
            self.iterator = iter(self.dataloader)
        else:
            self.iterator = None

    def new_batch(self):
        if self.iterator is not None:
            try:
                X_batch, Y_batch = next(self.iterator)
            except StopIteration:
                # Reinitialize the iterator if it reaches the end
                self.iterator = iter(self.dataloader)
                X_batch, Y_batch = next(self.iterator)
            return X_batch, Y_batch
        else:
            return self.X_all, self.Y_all

    def variational_loss(self, X, Y):
        K_mm = self.kernel(self.xm, self.xm) + JITTER * torch.eye(self.xm.size(0), dtype=torch.float64)
        Lm = torch.linalg.cholesky(K_mm)
        K_mm_inv = torch.cholesky_inverse(Lm)
        K_mn = self.kernel(self.xm, X)
        K_nm = K_mn.t()
        qu_S = self.chole @ self.chole.t() + JITTER * torch.eye(self.xm.size(0),
                                                                dtype=torch.float64)  # Ensure positive definite
        Ls = torch.linalg.cholesky(qu_S)
        K_nn = self.kernel(X, X).diag()
        batch_size = X.size(0)
        # K_nm * K_mm_inv * m, (b, 1)
        mean_vector = K_nm @ K_mm_inv @ self.qu_mean

        # diag(K_tilde), (b, 1)
        precision = 1 / self.log_beta.exp()
        K_tilde = precision * (K_nn - (K_nm @ K_mm_inv @ K_mn).diag())

        # k_i \cdot k_i^T, (b, m, m)
        # Expand dimensions and transpose for batch
        K_nm_expanded = K_nm.unsqueeze(2)  # Shape (b, m, 1)
        K_nm_transposed = K_nm_expanded.transpose(1, 2)  # Shape (b, 1, m)

        # Perform batch matrix multiplication
        lambda_mat = torch.matmul(K_nm_expanded, K_nm_transposed)  # Shape (b, m, m)
        # K_mm_inv \cdot k_i \cdot k_i^T \cdot K_mm_inv, (b, m, m)
        lambda_mat = K_mm_inv @ lambda_mat @ K_mm_inv
        # Trace terms, (b,)
        batch_matrices = qu_S @ lambda_mat
        traces = precision * torch.einsum('bii->b', batch_matrices)

        # Likelihood
        likelihood_sum = -0.5 * batch_size * torch.log(2 * torch.tensor(PI)) + 0.5 * batch_size * torch.log(
            self.log_beta.exp()) \
                         - 0.5 * self.log_beta.exp() * ((Y - K_nm @ K_mm_inv @ self.qu_mean) ** 2).sum(dim=0).view(-1,
                                                                                                                   1) - 0.5 * torch.sum(
            K_tilde) - 0.5 * torch.sum(traces)

        # Compute KL
        logdetS = 2 * Ls.diag().abs().log().sum()
        logdetKmm = 2 * Lm.diag().abs().log().sum()
        KL = 0.5 * (K_mm_inv @ qu_S).diag().sum(dim=0).view(-1, 1) + 0.5 * (self.qu_mean.t() @ K_mm_inv @ self.qu_mean) \
             - 0.5 * logdetS + 0.5 * logdetKmm - 0.5 * self.num_inducing

        return likelihood_sum, KL, mean_vector

    def forward(self, Xte):
        Xte=self.normalizer.normalize(Xte,'x')
        K_mm = self.kernel(self.xm, self.xm) + JITTER * torch.eye(self.xm.size(0), dtype=torch.float64)
        Lm = torch.linalg.cholesky(K_mm)
        K_mm_inv = torch.cholesky_inverse(Lm)
        K_tt = self.kernel(Xte, Xte)
        K_tm = self.kernel(Xte, self.xm)
        A = K_tm @ K_mm_inv  # (t, m)
        mean = A @ self.qu_mean  # (t, 1)

        #denormalize
        mean = self.normalizer.denormalize(mean,'y')

        return mean

if __name__ == '__main__':
    # Train and evaluate the model
    torch.manual_seed(4)
    # Train set
    num_data =10000
    xtr,ytr,xte,yte=data.generate(num_data,1000,seed=4,input_dim=18)



    # Training the model
    num_inducing =100
    batch_size = 1000
    learning_rate = 0.1
    num_epochs = 500

    # Create an instance of SVIGP
    model = SVIGP(xtr, ytr, num_inducing=num_inducing, batchsize=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(num_epochs):
        optimizer.zero_grad()
        X_batch, Y_batch = model.new_batch()
        likelihood, KL, _ = model.variational_loss(X_batch, Y_batch)

        loss = KL - likelihood * num_data / batch_size  # Negative ELBO
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print('iter', i, 'nll:{:.5f}'.format(loss.item()))

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        predictions = model(xte)
        mse = torch.mean((predictions - yte) ** 2)
        print(f'Test MSE: {mse.item()}')

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(xtr.numpy(), ytr.numpy(), 'kx', label='Training data')
    plt.plot(xte.numpy(), yte.numpy(), 'b', label='True function')
    plt.plot(xte.numpy(), predictions.numpy(), 'r', label='Predicted function')
    plt.legend()
    plt.show()