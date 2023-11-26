# A modular kernel implementation. The user just need to define the kernel function but not the hyperparameters. The kernel will automatically create the hyperparameters it needs and store them as nn.Parameter. GP will use the kernel to calculate the covariance matrix and the mean function to calculate the mean vector. The likelihood is tied to the GP and not spedified by the user. A GP model should have it own likelihood function by the user.
# 
# Author: Wei W. Xing (wxing.me)
# Email: wayne.xingle@gmail.com
# Date: 2023-11-26
# 
# 
# log
# 2023-11-26 v2.0.0
# use torch.cdist to calculate distance matrix. Introduce stationKernel to handel stationary kernel. by wei xing.
# To ensure the positive of the hyperparameters, we can use 
# 1. length_scales = torch.abs(self.length_scales) + self.eps. Good for input with bounded range.
# 2. length_scales = torch.exp(self.length_scales) + self.eps. Good for input grow exponentially.
# 3. length_scales = F.softplus(self.raw_length_scales) where F is torch.nn.functional.
# I choose No.1 for now by hoping that the data will be processed to be close the stationary and normal distribution.


import torch
import torch.nn as nn
EPS = 1e-9

class LinearKernel(nn.Module):
    def __init__(self, input_dim, initial_length_scale=1.0, initial_signal_variance=1.0):
        super(LinearKernel, self).__init__()
        self.length_scales = nn.Parameter(torch.ones(input_dim) * initial_length_scale)
        self.signal_variance = nn.Parameter(torch.tensor([initial_signal_variance]))
        self.center = nn.Parameter(torch.zeros(input_dim))
        
        
    def forward(self, x1, x2):
        x1 = x1 / self.length_scales - self.center
        x2 = x2 / self.length_scales - self.center
        
        return x1 @ x2.T * self.signal_variance.abs()

class ARDKernel(nn.Module):
    def __init__(self, input_dim, initial_length_scale=1.0, initial_signal_variance=1.0, eps=EPS):
        super().__init__()
        # Initialize length scales for each dimension
        self.length_scales = nn.Parameter(torch.ones(input_dim) * initial_length_scale)
        self.signal_variance = nn.Parameter(torch.tensor([initial_signal_variance]))
        self.eps = eps  # Small constant to prevent division by zero

    def forward(self, x1, x2):
        # Ensure length scales are positive and add a small constant for numerical stability
        length_scales = torch.abs(self.length_scales) + self.eps

        # Compute scaled squared Euclidean distances
        scaled_x1 = x1 / length_scales
        scaled_x2 = x2 / length_scales
        sqdist = torch.cdist(scaled_x1, scaled_x2, p=2)**2
        return self.signal_variance.abs() * torch.exp(-0.5 * sqdist)

    
# Matern kernel with independent length scales
class MaternKernel(nn.Module):
    def __init__(self, input_dim, initial_length_scale=1.0, initial_signal_variance=1.0, nu=2.5, eps=EPS):
        super().__init__()
        # Initialize length scales for each dimension
        self.length_scales = nn.Parameter(torch.ones(input_dim) * initial_length_scale)
        self.signal_variance = nn.Parameter(torch.tensor([initial_signal_variance]))
        self.nu = nn.Parameter(torch.tensor([nu]))
        self.eps = eps
    def forward(self, x1, x2):
        # Ensure length scales are positive and add a small constant for numerical stability
        length_scales = torch.abs(self.length_scales) + self.eps

        # Compute scaled squared Euclidean distances
        scaled_x1 = x1 / length_scales
        scaled_x2 = x2 / length_scales
        sqdist = torch.cdist(scaled_x1, scaled_x2, p=2)**2
        return self.signal_variance.abs() * torch.pow(1 + torch.sqrt(3 * sqdist) / length_scales.pow(2), -self.nu)
    

# kernel operations
class SumKernel(nn.Module):
    def __init__(self, kernel1, kernel2):
        super().__init__()
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def forward(self, x1, x2):
        return self.kernel1(x1, x2) + self.kernel2(x1, x2)

class ProductKernel(nn.Module):
    def __init__(self, kernel1, kernel2):
        super().__init__()
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def forward(self, x1, x2):
        return self.kernel1(x1, x2) * self.kernel2(x1, x2)

# deprecated kernels where length_scale is a scalar
class SquaredExponentialKernel(nn.Module):
    def __init__(self, length_scale=1.0, signal_variance=1.0):
        super().__init__()
        self.length_scale = nn.Parameter(torch.tensor([length_scale]))
        self.signal_variance = nn.Parameter(torch.tensor([signal_variance]))

    def forward(self, x1, x2):
        sqdist = torch.sum(x1**2, 1).reshape(-1, 1) + torch.sum(x2**2, 1) - 2 * torch.matmul(x1, x2.T)
        return self.signal_variance.pow(2) * torch.exp(-0.5 * sqdist / self.length_scale.pow(2))
    
    # need to check if this is correct
class RationalQuadraticKernel(nn.Module):
    def __init__(self, length_scale=1., signal_variance=1., alpha=1.):
        super(RationalQuadraticKernel, self).__init__()
        self.length_scale = nn.Parameter(torch.tensor([length_scale]))
        self.signal_variance = nn.Parameter(torch.tensor([signal_variance]))
        self.alpha = nn.Parameter(torch.tensor([alpha]))

    def forward(self, x1, x2):
        sqdist = torch.sum(x1**2, 1).reshape(-1, 1) + torch.sum(x2**2, 1) - 2 * torch.matmul(x1, x2.T)
        return self.signal_variance.pow(2) * torch.pow(1 + 0.5 * sqdist / self.alpha / self.length_scale.pow(2), -self.alpha)
  
class MaternKernel_scalarLengthScale(nn.Module):
    def __init__(self, length_scale=1.0, signal_variance=1.0, nu=2.5):
        super().__init__()
        self.length_scale = nn.Parameter(torch.tensor([length_scale]))
        self.signal_variance = nn.Parameter(torch.tensor([signal_variance]))
        self.nu = nn.Parameter(torch.tensor([nu]))

    def forward(self, x1, x2):
        sqdist = torch.sum(x1**2, 1).reshape(-1, 1) + torch.sum(x2**2, 1) - 2 * torch.matmul(x1, x2.T)
        return self.signal_variance.pow(2) * torch.pow(1 + torch.sqrt(3 * sqdist) / self.length_scale.pow(2), -self.nu)

        
