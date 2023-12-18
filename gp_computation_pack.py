# commonly used functions for GP computation
# Author: Wei Xing
# Date: 2023-12-11
# Version: 1.0
# History:
# 1.0    2023-12-11    Initial version

import torch
import torch.nn as nn
import numpy as np


EPS = 1e-9

# define a normalization module
class Normalize0_layer(nn.Module):
    # special normalization, i.e., all dimensions are normalized together. This works well for conditional independent GP (CIGP) for normalizing y.
    def __init__(self, X0, if_trainable =False):
        super().__init__()
        self.mean = nn.Parameter(X0.mean(), requires_grad=if_trainable)
        self.std = nn.Parameter(X0.std(), requires_grad=if_trainable)
    def forward(self, x):
        return (x - self.mean) / self.std
    def inverse(self, x):
        return x * self.std + self.mean
    
class Normalize_layer(nn.Module):
    # normal normalization. It is basically the pytorch batch normalization, but the mean and std are not trainable. 
    # It work well for normalizing the input x.
    def __init__(self, X0, dim=0, if_trainable =False):
        super().__init__()
        self.mean = nn.Parameter(X0.mean(dim), requires_grad=if_trainable)
        self.std = nn.Parameter(X0.std(dim), requires_grad=if_trainable)
    def forward(self, x):
        return (x - self.mean) / self.std
    def inverse(self, x):
        return x * self.std + self.mean
    
# TODO: add a warpping layer. follow https://botorch.org/tutorials/bo_with_warped_gp 
# class warp_layer(nn.Module):
#     def __init__(self, warp_func, if_trainable =False):
#         super().__init__()
#         self.warp_func = warp_func
#         self.warp_func.requires_grad = if_trainable
#     def forward(self, x):
#         return self.warp_func(x)
#     def inverse(self, x):
#         return self.warp_func.inverse(x)


class cdf(nn.Module):
    def __init__(self,concentration1_prior, concentration0_prior):
        super().__init__()
        self.concentration1_prior=concentration1_prior
        self.concentration0_prior=concentration0_prior
        self.low = torch.full_like(self.concentration0, 0)
        self.high = torch.full_like(self.concentration0, 1)

    def forward(self, x):
        x= -1 * (x.clamp(0, 1) - 0.5) + 0.5
        x = x * (self.high - self.low) + self.low
        x= x.pow(self.concentration0_prior.reciprocal())
        x= 1 - x
        wrap_x=x.pow(self.concentration1_prior.reciprocal())
        return wrap_x

    def inverse(self, y):
        y= y.pow(1 / self.concentration1_prior.reciprocal())
        y= 1 - y
        y= y.pow(1 / self.concentration0_prior.reciprocal())
        y= -1 * (y - 0.5) + 0.5
        return y

class WarpLayer(nn.Module):
    def __init__(self, warp_func, if_trainable=False):
        super(WarpLayer, self).__init__()
        self.warp_func = warp_func
        self.if_trainable = if_trainable

        if self.if_trainable:
            for param in self.warp_func.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.warp_func(x)

    def inverse(self, y):
        return self.warp_func.inverse(y)


# compute the log likelihood of a normal distribution
def Gaussian_log_likelihood(y, cov, Kinv_method='cholesky3'):
    """
    Compute the log-likelihood of a Gaussian distribution.

    Args:
        y (torch.Tensor): The observed values.
        mean (torch.Tensor): The mean of the Gaussian distribution.
        cov (torch.Tensor): The covariance matrix of the Gaussian distribution.
        Kinv_method (str, optional): The method to compute the inverse of the covariance matrix.
            Defaults to 'cholesky3'.

    Returns:
        torch.Tensor: The log-likelihood of the Gaussian distribution.

    Raises:
        ValueError: If Kinv_method is not 'direct' or 'cholesky'.
    """
    
    # assert if the correct dimension
    assert len(y.shape) == 2 and len(cov.shape) == 2, "y, mean, cov should be 2D tensors"
    
    if Kinv_method == 'cholesky1':
        L = torch.cholesky(cov)
        L_inv = torch.inverse(L)
        K_inv = L_inv.T @ L_inv
        return -0.5 * (y.T @ K_inv @ y + torch.logdet(cov) + len(y) * np.log(2 * np.pi))
    elif Kinv_method == 'cholesky2':
        L = torch.cholesky(cov)
        return -0.5 * (y.T @ torch.cholesky_solve(y, L) + torch.logdet(cov) + len(y) * np.log(2 * np.pi))
    
    elif Kinv_method == 'cholesky3':
        # fastest implementation so far
        L = torch.cholesky(cov)
        # return -0.5 * (y_use.T @ torch.cholesky_solve(y_use, L) + L.diag().log().sum() + len(x_train) * np.log(2 * np.pi))
        if y.shape[1] > 1:
            Warning('y_use.shape[1] > 1, will treat each column as a sample (for the joint normal distribution) and sum the log-likelihood')
            # 
            # (Alpha ** 2).sum() = (Alpha @ Alpha^T).diag().sum() = \sum_i (Alpha @ Alpha^T)_{ii}
            # 
            y_dim = y.shape[1]
            log_det_K = 2 * torch.sum(torch.log(torch.diag(L)))
            Alpha = torch.cholesky_solve(y, L, upper = False)
            return - 0.5 * ( (Alpha ** 2).sum() + log_det_K * y_dim + len(y) * y_dim * np.log(2 * np.pi) )
        else:
            return -0.5 * (y.T @ torch.cholesky_solve(y, L) + L.diag().log().sum() + len(y) * np.log(2 * np.pi))

    elif Kinv_method == 'direct':
        K_inv = torch.inverse(cov)
        return -0.5 * (y.T @ K_inv @ y + torch.logdet(cov) + len(y) * np.log(2 * np.pi))
    elif Kinv_method == 'torch_distribution_MN1':
        L = torch.cholesky(cov)
        return torch.distributions.MultivariateNormal(y, scale_tril=L).log_prob(y)
    elif Kinv_method == 'torch_distribution_MN2':
        return torch.distributions.MultivariateNormal(y, cov).log_prob(y)
    else:
        raise ValueError('Kinv_method should be either direct or cholesky')
    
def conditional_Gaussian(y, Sigma, K_s, K_ss, Kinv_method='cholesky3'):
    # Sigma = Sigma + torch.eye(len(Sigma)) * EPS
    if Kinv_method == 'cholesky1':   # kernel inverse is not stable, use cholesky decomposition instead
        L = torch.cholesky(Sigma)
        L_inv = torch.inverse(L)
        K_inv = L_inv.T @ L_inv
        alpha = K_inv @ y
        mu = K_s.T @ alpha
        v = L_inv @ K_s
        cov = K_ss - v.T @ v
    elif Kinv_method == 'cholesky3':
        # recommended implementation, fastest so far
        L = torch.cholesky(Sigma)
        alpha = torch.cholesky_solve(y, L)
        mu = K_s.T @ alpha
        # v = torch.cholesky_solve(K_s, L)    # wrong implementation
        v = L.inverse() @ K_s   # correct implementation
        cov = K_ss - v.T @ v
    elif Kinv_method == 'direct':
        K_inv = torch.inverse(Sigma)
        mu = K_s.T @ K_inv @ y
        cov = K_ss - K_s.T @ K_inv @ K_s
    else:
        raise ValueError('Kinv_method should be either direct or cholesky')
    
    cov = cov.diag().view(-1, 1).expand_as(mu) ##important!!!
    return mu, cov