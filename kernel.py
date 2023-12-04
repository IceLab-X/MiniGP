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
# 

import torch
import torch.nn as nn

EPS = 1e-9

class LinearKernel(nn.Module):
    """
    Linear kernel module.

    Args:
        input_dim (int): The input dimension.
        initial_length_scale (float): The initial length scale value. Default is 1.0.
        initial_signal_variance (float): The initial signal variance value. Default is 1.0.

    Attributes:
        length_scales (nn.Parameter): The length scales for each dimension.
        signal_variance (nn.Parameter): The signal variance.
        center (nn.Parameter): The center.

    """

    def __init__(self, input_dim, initial_length_scale=1.0, initial_signal_variance=1.0):
        super(LinearKernel, self).__init__()
        self.length_scales = nn.Parameter(torch.ones(input_dim) * initial_length_scale)
        self.signal_variance = nn.Parameter(torch.tensor([initial_signal_variance]))
        self.center = nn.Parameter(torch.zeros(input_dim))
        
    def forward(self, x1, x2):
        """
        Compute the covariance matrix using the linear kernel.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The covariance matrix.

        """
        x1 = x1 / self.length_scales - self.center
        x2 = x2 / self.length_scales - self.center
        
        return x1 @ x2.T * self.signal_variance.abs()

class ARDKernel(nn.Module):
    """
    ARD (Automatic Relevance Determination) kernel module.

    Args:
        input_dim (int): The input dimension.
        initial_length_scale (float): The initial length scale value. Default is 1.0.
        initial_signal_variance (float): The initial signal variance value. Default is 1.0.
        eps (float): A small constant to prevent division by zero. Default is 1e-9.

    Attributes:
        length_scales (nn.Parameter): The length scales for each dimension.
        signal_variance (nn.Parameter): The signal variance.
        eps (float): A small constant to prevent division by zero.

    """

    def __init__(self, input_dim, initial_length_scale=1.0, initial_signal_variance=1.0, eps=EPS):
        super().__init__()
        self.length_scales = nn.Parameter(torch.ones(input_dim) * initial_length_scale)
        self.signal_variance = nn.Parameter(torch.tensor([initial_signal_variance]))
        self.eps = eps

    def forward(self, x1, x2):
        """
        Compute the covariance matrix using the ARD kernel.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The covariance matrix.

        """
        length_scales = torch.abs(self.length_scales) + self.eps

        scaled_x1 = x1 / length_scales
        scaled_x2 = x2 / length_scales
        sqdist = torch.cdist(scaled_x1, scaled_x2, p=2)**2
        return self.signal_variance.abs() * torch.exp(-0.5 * sqdist)

    
# Matern kernel with independent length scales
class MaternKernel(nn.Module):
    """
    Matern kernel module with independent length scales.

    Args:
        input_dim (int): The input dimension.
        initial_length_scale (float): The initial length scale value. Default is 1.0.
        initial_signal_variance (float): The initial signal variance value. Default is 1.0.
        nu (float): The smoothness parameter. Default is 2.5.
        eps (float): A small constant to prevent division by zero. Default is 1e-9.

    Attributes:
        length_scales (nn.Parameter): The length scales for each dimension.
        signal_variance (nn.Parameter): The signal variance.
        nu (nn.Parameter): The smoothness parameter.
        eps (float): A small constant to prevent division by zero.

    """

    def __init__(self, input_dim, initial_length_scale=1.0, initial_signal_variance=1.0, nu=2.5, eps=EPS):
        super().__init__()
        self.length_scales = nn.Parameter(torch.ones(input_dim) * initial_length_scale)
        self.signal_variance = nn.Parameter(torch.tensor([initial_signal_variance]))
        self.nu = nn.Parameter(torch.tensor([nu]))
        self.eps = eps

    def forward(self, x1, x2):
        """
        Compute the covariance matrix using the Matern kernel.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The covariance matrix.

        """
        length_scales = torch.abs(self.length_scales) + self.eps

        scaled_x1 = x1 / length_scales
        scaled_x2 = x2 / length_scales
        sqdist = torch.cdist(scaled_x1, scaled_x2, p=2)**2
        return self.signal_variance.abs() * torch.pow(1 + torch.sqrt(3 * sqdist) / length_scales.pow(2), -self.nu)
    

# kernel operations
class SumKernel(nn.Module):
    """
    Sum of two kernels module.

    Args:
        kernel1 (nn.Module): The first kernel.
        kernel2 (nn.Module): The second kernel.

    Attributes:
        kernel1 (nn.Module): The first kernel.
        kernel2 (nn.Module): The second kernel.

    """

    def __init__(self, kernel1, kernel2):
        super().__init__()
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def forward(self, x1, x2):
        """
        Compute the covariance matrix using the sum of two kernels.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The covariance matrix.

        """
        return self.kernel1(x1, x2) + self.kernel2(x1, x2)

class ProductKernel(nn.Module):
    """
    Product of two kernels module.

    Args:
        kernel1 (nn.Module): The first kernel.
        kernel2 (nn.Module): The second kernel.

    Attributes:
        kernel1 (nn.Module): The first kernel.
        kernel2 (nn.Module): The second kernel.

    """

    def __init__(self, kernel1, kernel2):
        super().__init__()
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def forward(self, x1, x2):
        """
        Compute the covariance matrix using the product of two kernels.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The covariance matrix.

        """
        return self.kernel1(x1, x2) * self.kernel2(x1, x2)

# deprecated kernels where length_scale is a scalar
class SquaredExponentialKernel(nn.Module):
    """
    Squared Exponential kernel module with scalar length scale.

    Args:
        length_scale (float): The length scale value. Default is 1.0.
        signal_variance (float): The signal variance value. Default is 1.0.

    Attributes:
        length_scale (nn.Parameter): The length scale.
        signal_variance (nn.Parameter): The signal variance.

    """

    def __init__(self, length_scale=1.0, signal_variance=1.0):
        super().__init__()
        self.length_scale = nn.Parameter(torch.tensor([length_scale]))
        self.signal_variance = nn.Parameter(torch.tensor([signal_variance]))

    def forward(self, x1, x2):
        """
        Compute the covariance matrix using the squared exponential kernel.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The covariance matrix.

        """
        sqdist = torch.sum(x1**2, 1).reshape(-1, 1) + torch.sum(x2**2, 1) - 2 * torch.matmul(x1, x2.T)
        return self.signal_variance.pow(2) * torch.exp(-0.5 * sqdist / self.length_scale.pow(2))
    
    # need to check if this is correct
class RationalQuadraticKernel(nn.Module):
    """
    Rational Quadratic kernel module with scalar length scale.

    Args:
        length_scale (float): The length scale value. Default is 1.0.
        signal_variance (float): The signal variance value. Default is 1.0.
        alpha (float): The alpha value. Default is 1.0.

    Attributes:
        length_scale (nn.Parameter): The length scale.
        signal_variance (nn.Parameter): The signal variance.
        alpha (nn.Parameter): The alpha value.

    """

    def __init__(self, length_scale=1., signal_variance=1., alpha=1.):
        super(RationalQuadraticKernel, self).__init__()
        self.length_scale = nn.Parameter(torch.tensor([length_scale]))
        self.signal_variance = nn.Parameter(torch.tensor([signal_variance]))
        self.alpha = nn.Parameter(torch.tensor([alpha]))

    def forward(self, x1, x2):
        """
        Compute the covariance matrix using the rational quadratic kernel.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The covariance matrix.

        """
        sqdist = torch.sum(x1**2, 1).reshape(-1, 1) + torch.sum(x2**2, 1) - 2 * torch.matmul(x1, x2.T)
        return self.signal_variance.pow(2) * torch.pow(1 + 0.5 * sqdist / self.alpha / self.length_scale.pow(2), -self.alpha)
  
class MaternKernel_scalarLengthScale(nn.Module):
    """
    Matern kernel module with scalar length scale.

    Args:
        length_scale (float): The length scale value. Default is 1.0.
        signal_variance (float): The signal variance value. Default is 1.0.
        nu (float): The smoothness parameter. Default is 2.5.

    Attributes:
        length_scale (nn.Parameter): The length scale.
        signal_variance (nn.Parameter): The signal variance.
        nu (nn.Parameter): The smoothness parameter.

    """

    def __init__(self, length_scale=1.0, signal_variance=1.0, nu=2.5):
        super().__init__()
        self.length_scale = nn.Parameter(torch.tensor([length_scale]))
        self.signal_variance = nn.Parameter(torch.tensor([signal_variance]))
        self.nu = nn.Parameter(torch.tensor([nu]))

    def forward(self, x1, x2):
        """
        Compute the covariance matrix using the Matern kernel with scalar length scale.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The covariance matrix.

        """
        sqdist = torch.sum(x1**2, 1).reshape(-1, 1) + torch.sum(x2**2, 1) - 2 * torch.matmul(x1, x2.T)
        return self.signal_variance.pow(2) * torch.pow(1 + torch.sqrt(3 * sqdist) / self.length_scale.pow(2), -self.nu)

        
