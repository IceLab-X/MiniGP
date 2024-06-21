import torch
import torch.nn as nn

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

class MaternKernel(nn.Module):
    def __init__(self, length_scale=1.0, signal_variance=1.0, nu=2.5):
        super().__init__()
        self.length_scale = nn.Parameter(torch.tensor([length_scale]))
        self.signal_variance = nn.Parameter(torch.tensor([signal_variance]))
        self.nu = nn.Parameter(torch.tensor([nu]))

    def forward(self, x1, x2):
        sqdist = torch.sum(x1**2, 1).reshape(-1, 1) + torch.sum(x2**2, 1) - 2 * torch.matmul(x1, x2.T)
        return self.signal_variance.pow(2) * torch.pow(1 + torch.sqrt(3 * sqdist) / self.length_scale.pow(2), -self.nu)

        
