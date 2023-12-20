# gp_transform.py
# transform block for GP models
# Compared to the standard transform in torch, this transform also takes care of the covariance matrix.
# 
# Author: Wei Xing
# Date: 2023-12-13
# Version: 1.0
# History:
# 1.0    2023-12-13    Initial version

import torch
import torch.nn as nn

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

# TODO: Not finished yet, this is just a draft. TO finish, see, https://en.wikipedia.org/wiki/Beta_distribution and https://botorch.org/tutorials/bo_with_warped_gp.
class BetaCDF_Warp_Layer(nn.Module):
    # warp layer for GP models.
    # Warming: input x should be normalized to [0, 1] before using this layer.
    def __init__(self, dim, a0 = 1, b0 = 1):
        super(BetaCDF_Warp_Layer, self).__init__()
        self.a = nn.Parameter(torch.ones(dim) * a0)
        self.b = nn.Parameter(torch.ones(dim) * b0)

    def forward(self, x):
        # use the beta cdf to warp the input x
        # return torch.distributions.beta.Beta(self.a, self.b).cdf(x)
        # 
        # OR explicitly compute the cdf
        return x.pow(self.a.abs() - 1) * (1 - x).pow(self.b - 1) / torch.special.beta(self.a, self.b)
        # TODO: need to check why torch.special.beta(self.a, self.b) not working 
        # return x.pow(self.a.abs() - 1.) * (1 - x).pow(self.b.abs() - 1.) / torch.distributions.beta.Beta(self.a.abs(), self.b.abs()).mean
        

    def inverse(self, y):
        #  inverse the beta cdf to get the original x
        # TODO: need to test the capcity for vector input
        return torch.distributions.beta.Beta(self.a.abs(), self.b.abs()).icdf(y)