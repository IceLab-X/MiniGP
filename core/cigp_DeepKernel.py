
# Conditional independent Gaussian process (CIGP) for vector output regression based on pytorch
# 
# CIGP use a single kernel for each output. Thus the log likelihood is simply a sum of the log likelihood of each output.

# Author: Wei W. Xing (wxing.me)
# Email: wayne.xingle@gmail.com
# Date: 2023-11-26
from data_sample import generate_example_data as data
import numpy as np
import torch
import torch.nn as nn
from core.cigp_v10_merge_with_cigp_baseline import cigp
import core.GP_CommonCalculation as gp_pack
import matplotlib.pyplot as plt
EPS = 1e-10


class CIGP_DKL(nn.Module):
    def __init__(self, X,Y, normal_y_mode=0):
        super().__init__()
        # normalize X independently for each dimension
        self.X=X
        self.Y=Y

        # GP hyperparameters
        self.log_beta = nn.Parameter(
            torch.ones(1) * 0)  # a large noise by default. Smaller value makes larger noise variance.
        self.log_length_scale = nn.Parameter(torch.zeros(3))  # ARD length scale
        self.log_scale = nn.Parameter(torch.zeros(1))  # kernel scale
        input_dim = self.X.shape[1]

        self.FeatureExtractor = torch.nn.Sequential(nn.Linear(input_dim, input_dim*2),
            nn.LeakyReLU(),
            nn.Linear(input_dim *2, input_dim *2),
            nn.LeakyReLU(),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(input_dim * 2, input_dim))
        

    def kernel(self, X1, X2):
        # the common RBF kernel
        X1 = X1 / self.log_length_scale.exp()
        X2 = X2 / self.log_length_scale.exp()
        # X1_norm2 = X1 * X1
        # X2_norm2 = X2 * X2
        X1_norm2 = torch.sum(X1 * X1, dim=1).view(-1, 1)
        X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)

        K = -2.0 * X1 @ X2.t() + X1_norm2.expand(X1.size(0), X2.size(0)) + X2_norm2.t().expand(X1.size(0), X2.size(0))  #this is the effective Euclidean distance matrix between X1 and X2.
        K = self.log_scale.exp() * torch.exp(-0.5 * K)
        return K

    def forward(self, x_test):

        
        x_train = self.FeatureExtractor(self.X)
        x_test = self.FeatureExtractor(x_test)
        
        K = self.kernel(x_train, x_train) + self.log_beta.exp().pow(-1) * torch.eye(self.X.shape[0])
        K_s = self.kernel(x_train, x_test)
        K_ss = self.kernel(x_test, x_test)

        
        mu, cov = gp_pack.conditional_Gaussian(self.Y, K, K_s, K_ss)
        cov = cov.sum(dim=0).view(-1, 1)+ self.log_beta.exp().pow(-1)

        
        return mu, cov

    def negative_log_likelihood(self):

        
        x_train = self.FeatureExtractor(self.X)
        K = self.kernel(x_train, x_train) + self.log_beta.exp().pow(-1) * torch.eye(self.X.shape[0])
        
        return -gp_pack.Gaussian_log_likelihood(self.Y, K)
        # modified by Wei Xing to penalize the variance term 
        # return gp_pack.Gaussian_log_likelihood(y_train - mean_part_train, K) / x_train.shape[0] - torch.log(self.noise_variance) * 2
    
    def train_adam(self, niteration=10, lr=0.1):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        for i in range(niteration):
            optimizer.zero_grad()
            loss = self.negative_log_likelihood()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('iter %d, loss %.3f' % (i, loss.item()))
        return loss

    def print_parameters(self):
        for name, param in self.named_parameters():
            print(f"Parameter name: {name}, shape: {param.shape}")
xtr, ytr,xte,yte = data.generate(600,100,seed=42,input_dim=3)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
xtr, ytr, xte, yte = data.generate(600, 100, seed=42, input_dim=3)
model = CIGP_DKL(xtr, ytr)
model2 = cigp(xtr, ytr)

# Initialize lists to store R^2 values
r2_values_model = []
r2_values_model2 = []

# Training function modified to track R^2 scores
def train_model_with_r2_tracking(model, xte, yte, epochs, lr, r2_values):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.negative_log_likelihood()
        loss.backward()
        optimizer.step()

        # Calculate R^2 score with no gradient calculation
        with torch.no_grad():
            y_pred, _ = model.forward(xte)
            r2 = r2_score(yte.numpy(), y_pred.numpy())
            r2_values.append(r2)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}, R^2: {r2}")

# Train both models and track R^2
train_model_with_r2_tracking(model, xte, yte, 800, 0.1, r2_values_model)
train_model_with_r2_tracking(model2, xte, yte, 800, 0.1, r2_values_model2)

# Plotting the R^2 values
plt.figure(figsize=(10, 6))
plt.plot(range(800), r2_values_model, label='Model 1 (CIGP_DKL)')
plt.plot(range(800), r2_values_model2, label='Model 2 (cigp)')
plt.xlabel('Training Epochs')
plt.ylabel('R^2 Score')
plt.title('R^2 Score vs. Training Epochs')
plt.legend()
plt.show()