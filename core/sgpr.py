#Author: Zidong Chen
#Date: 2024-06-22
# This is the implementation of the variational sparse Gaussian process (VSGP) model.
# More details can be found in the paper "Variational Learning of Inducing Variables in Sparse Gaussian Processes" by Titsias (2009).

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from model_FAQ.non_positive_definite_fixer import remove_similar_data
import core.GP_CommonCalculation as GP
from data_sample import generate_example_data as data
from core.kernel import ARDKernel
from core.cigp_v10 import cigp
print(torch.__version__)
# I use torch (1.11.0) for this work. lower version may not work.

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Fixing strange error if run in MacOS
JITTER = 1e-3
EPS = 1e-10
PI = 3.1415


class vsgp(nn.Module):
    def __init__(self, X, Y, num_inducing, normal_y_mode=0):
        super(vsgp, self).__init__()

        self.data = GP.data_normalization(X, Y, normal_y_mode)
        self.X, self.Y = self.data.normalize(X, Y)
        # normalize X independently for each dimension
        #self.X, self.Y = X, Y
        self.X, self.Y = remove_similar_data(self.X, self.Y)

        # GP hyperparameters
        self.log_beta = nn.Parameter(torch.ones(1) * -4)  # Initial noise level


        # Inducing points
        #subset_indices = torch.randperm(self.X.size(0))[:num_inducing]
        #self.xm = nn.Parameter(self.X[subset_indices])  # Inducing points
        input_dim=self.X.size(1)
        self.kernel = ARDKernel(input_dim)
        self.xm = nn.Parameter(torch.rand((num_inducing, input_dim)))  # Inducing points

    def negative_lower_bound(self):
        """Negative lower bound as the loss function to minimize."""
        n = self.X.size(0)
        K_mm = self.kernel(self.xm, self.xm) + JITTER * torch.eye(self.xm.size(0))
        L = torch.linalg.cholesky(K_mm)
        K_mn = self.kernel(self.xm, self.X)
        K_nn = self.kernel(self.X, self.X)
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
        K_mm = self.kernel(self.xm, self.xm) + JITTER * torch.eye(self.xm.size(0))
        L = torch.linalg.cholesky(K_mm)
        L_inv = torch.inverse(L)
        K_mm_inv = L_inv.t() @ L_inv

        K_mn = self.kernel(self.xm, self.X)
        K_nm = K_mn.t()
        sigma = torch.inverse(K_mm + self.log_beta.exp() * K_mn @ K_nm)

        mean_m = self.log_beta.exp() * (K_mm @ sigma @ K_mn) @ self.Y
        A_m = K_mm @ sigma @ K_mm
        return mean_m, A_m, K_mm_inv

    def forward(self, Xte):
        """Compute mean and variance for posterior distribution."""
        Xte = self.data.normalize(Xte)

        K_tt = self.kernel(Xte, Xte)
        K_tm = self.kernel(Xte, self.xm)
        K_mt = K_tm.t()
        mean_m, A_m, K_mm_inv = self.optimal_inducing_point()
        mean = (K_tm @ K_mm_inv) @ mean_m
        var = (K_tt - K_tm @ K_mm_inv @ K_mt +
               K_tm @ K_mm_inv @ A_m @ K_mm_inv @ K_mt)
        var_diag = var.diag().view(-1, 1)
        mean, var_diag = self.data.denormalize_result(mean, var_diag)
        # de-normalized

        return mean, var_diag

    def train_adam(self, niteration=10, lr=0.1):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        for i in range(niteration):
            optimizer.zero_grad()
            loss = self.negative_lower_bound()
            loss.backward()
            optimizer.step()
            print('iter', i, ' nll:', loss.item())

    def train_lbfgs(self, max_iter=20,lr=0.3):
        """Train model using LBFGS optimizer."""
        optimizer = torch.optim.LBFGS(self.parameters(), max_iter=max_iter,lr=lr)

        def closure():
            optimizer.zero_grad()
            loss = self.negative_lower_bound()
            loss.backward()  # Retain the graph
            print(self.log_length_scale.exp().item(), self.log_scale.exp().item())
            print(loss)
            return loss

        optimizer.step(closure)



def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
def mse_cal(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

xtr, ytr, xte, yte = data.generate(800, 100, seed=42, input_dim=3)
input_dim = xtr.shape[1]
#model2 = cigp(xtr, ytr)

# Initialize lists to store R^2 values
r2_values_model = []
r2_values_model2 = []


# Function to train the model and track MSE
def train_model_with_mse_tracking(model, xte, yte, epochs, lr, mse):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_mse = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.negative_lower_bound()
        loss.backward()
        optimizer.step()

        # Calculate MSE with no gradient calculation
        with torch.no_grad():
            y_pred, _ = model.forward(xte)
            mse_value = mse_cal(yte.numpy(), y_pred.numpy())
            mse.append(mse_value)

            if mse_value < best_mse:
                best_mse = mse_value
                best_state = model.state_dict()  # Save the best model state
            elif mse_value > best_mse and mse_value < 1:
                print(f"Stopping at epoch {epoch}/{epochs} with best MSE: {best_mse}")
                model.load_state_dict(best_state)  # Restore the best model state
                break

        if epoch % 1 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}, MSE: {mse[-1]}")

# Initialize MSE list
mse = []

# Train the model and track MSE
model = vsgp(xtr, ytr, 80)
model2= cigp(xtr, ytr)
train_model_with_mse_tracking(model, xte, yte, 400, 0.01, mse)
model2.train_adam(400, 0.01)
with torch.no_grad():

    y_pred2, _ = model2.forward(xte)
    mse2= mse_cal(yte.numpy(), y_pred2.numpy())
    print(f"Final MSE for model 2: {mse2}")
# Plotting the MSE values
plt.figure(figsize=(10, 6))
plt.plot(range(len(mse)), mse, label='Model 1 (vsgp)')
plt.xlabel('Training Epochs')
plt.ylabel('MSE')
plt.title('MSE Score vs. Training Epochs')
plt.legend()
plt.show()

# Plot predictions with error bars
