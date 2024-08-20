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
from core.kernel import ARDKernel
import core.GP_CommonCalculation as gp_pack

import matplotlib.pyplot as plt

EPS = 1e-10


class CIGP_DKL(nn.Module):
    def __init__(self, X, Y, normal_y_mode=0):
        super().__init__()
        # normalize X independently for each dimension
        self.normalizer = gp_pack.DataNormalization()
        self.normalizer.fit(X, 'x')
        self.normalizer.fit(Y, 'y')
        self.X = self.normalizer.normalize(X, 'x')
        self.Y = self.normalizer.normalize(Y, 'y')

        # GP hyperparameters
        self.log_beta = nn.Parameter(
            torch.ones(1) * 0)  # a large noise by default. Smaller value makes larger noise variance.

        input_dim = self.X.shape[1]
        self.kernel = ARDKernel(input_dim=input_dim)
        self.FeatureExtractor = torch.nn.Sequential(nn.Linear(input_dim, input_dim * 2),
                                                    nn.LeakyReLU(),
                                                    nn.Linear(input_dim * 2, input_dim * 2),
                                                    nn.LeakyReLU(),
                                                    nn.Linear(input_dim * 2, input_dim * 2),
                                                    nn.LeakyReLU(),
                                                    nn.Linear(input_dim * 2, input_dim))

    def forward(self, x_test):

        x_train = self.FeatureExtractor(self.X)
        x_test = self.FeatureExtractor(x_test)

        K = self.kernel(x_train, x_train) + self.log_beta.exp().pow(-1) * torch.eye(self.X.shape[0])
        K_s = self.kernel(x_train, x_test)
        K_ss = self.kernel(x_test, x_test)

        mu, cov = gp_pack.conditional_Gaussian(self.Y, K, K_s, K_ss)
        cov = cov.sum(dim=0).view(-1, 1) + self.log_beta.exp().pow(-1)

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
