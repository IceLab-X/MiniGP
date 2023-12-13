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
        x1 = (x1 - self.center) / self.length_scales
        x2 = (x2 - self.center) / self.length_scales
        
        # x1 = x1 / (self.length_scales - self.center)
        # x2 = x2 / (self.length_scales - self.center)
        
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
    Simplified Matern kernel module with independent length scales.
    For the full Matern kernel, see https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function 

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

    def __init__(self, input_dim, initial_length_scale=1.0, initial_signal_variance=1.0, nu=2.5, rho=1, eps=EPS):
        super().__init__()
        self.length_scales = nn.Parameter(torch.ones(input_dim) * initial_length_scale)
        self.signal_variance = nn.Parameter(torch.tensor([initial_signal_variance]))
        # self.nu = nn.Parameter(torch.tensor([nu]))    # not learnable but it can be learnable
        self.eps = eps
        self.nu = nu
        self.rho = rho

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
        # sqdist = torch.sum(scaled_x1**2, 1).reshape(-1, 1) + torch.sum(scaled_x2**2, 1) - 2 * torch.matmul(scaled_x1, scaled_x2.T)
        
        # if self.nu == 0.5:
        #     return self.signal_variance.abs() * torch.exp(-torch.sqrt(sqdist))
        # elif self.nu == 1.5:
        #     return self.signal_variance.abs() * (1 + torch.sqrt(3 * sqdist)) * torch.exp(-torch.sqrt(3 * sqdist))
        # elif self.nu == 2.5:
        #     return self.signal_variance.abs() * (1 + torch.sqrt(5 * sqdist) + 5 / 3 * sqdist) * torch.exp(-torch.sqrt(5 * sqdist))
        
        if self.nu == 0.5:
            return self.signal_variance.abs() * torch.exp(-torch.sqrt(sqdist)/self.rho)
        elif self.nu == 1.5:
            return self.signal_variance.abs() * (1 + torch.sqrt(3 * sqdist)/self.rho) * torch.exp(-torch.sqrt(3 * sqdist)/self.rho)
        elif self.nu == 2.5:
            return self.signal_variance.abs() * (1 + torch.sqrt(5 * sqdist)/self.rho + 5 / 3 * sqdist/self.rho**2) * torch.exp(-torch.sqrt(5 * sqdist)/self.rho)

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

     Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The covariance matrix.

        """
        sqdist = torch.sum(x1**2, 1).reshape(-1, 1) + torch.sum(x2**2, 1) - 2 * torch.matmul(x1, x2.T)
        return self.signal_variance.pow(2) * torch.pow(1 + torch.sqrt(3 * sqdist) / self.length_scale.pow(2), -self.nu)

class RBFKernel(nn.Module):
    """
    Radial Basis Function kernel module with scalar length scale.

    The RBF kernel is defined as:
        K(x, x') = σ² exp(-||x - x'||² / (2l²))
    where:
        σ² is the signal variance,
        l is the length scale,
        ||x - x'|| is the Euclidean distance between the input vectors.

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
        Compute the covariance matrix using the RBF kernel.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The covariance matrix.

        """
        x1_sq = torch.sum(x1 ** 2, 1).reshape(-1, 1)
        x2_sq = torch.sum(x2 ** 2, 1).reshape(1, -1)
        sqdist = x1_sq + x2_sq - 2 * torch.matmul(x1, x2.T)
        return self.signal_variance.pow(2) * torch.exp(-0.5 / self.length_scale.pow(2) * sqdist)


class PERKernel(nn.Module):
    """
    Periodic kernel (exp-sine-squared) module with scalar length scale, signal variance, and period.

    The kernel is defined as:
        K(x, x') = σ² exp(-2 sin²(π|x - x'| / p) / l²)
    where:
        σ² is the signal variance,
        l is the length scale,
        p is the period of the kernel,
        |x - x'| is the absolute difference between inputs.

    Args:
        length_scale (float): The length scale value (l). Default is 1.0.
        signal_variance (float): The signal variance value (σ²). Default is 1.0.
        period (float): The period of the kernel (p). Default is 1.0.

    Attributes:
        length_scale (nn.Parameter): The length scale (l).
        signal_variance (nn.Parameter): The signal variance (σ²).
        period (nn.Parameter): The period (p).

    """

    def __init__(self, length_scale=1.0, signal_variance=1.0, period=1.0):
        super().__init__()
        self.length_scale = nn.Parameter(torch.tensor([length_scale]))
        self.signal_variance = nn.Parameter(torch.tensor([signal_variance]))
        self.period = nn.Parameter(torch.tensor([period]))

    def forward(self, x1, x2):
        """
        Compute the covariance matrix using the periodic kernel.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The covariance matrix.

        """
        dist = torch.abs(x1.unsqueeze(1) - x2.unsqueeze(0))
        sin_term = torch.sin(math.pi * dist / self.period).pow(2)
        return self.signal_variance.pow(2) * torch.exp(-2 * sin_term / self.length_scale.pow(2))


class Neural_kernel(nn.Module):
    """
    Neural kernel module with scalar length scale.

    Args:
        RBF: length_scale ; signal_variance; (float) 
        Linear: length_scales; signal_variance; center; (float)
        RQ: lengthscale ; signal_variance ; alpha; (float)
        PER: length_scale ; signal_variance; period; (float)
        input_dim: the input dim of input vector.
        base_kernel_num: the number of base kernel. Default is 3.(The number and the kind of base kernel can change as you need)
    Attributes:
        RBF: length_scale (nn.Parameter): The length scale value.
        Linear: variance (nn.Parameter) : A variance parameter.
        RQ: lengthscale and alpha (nn.Parameter): The lengthscale parameter. The rational quadratic relative weighting parameter.

    """
    def __init__(self, input_dim, base_kernel_num=3):
        super(Neural_kernel, self).__init__()
        self.RBFKernel = RBFKernel()
        self.Linekern = LinearKernel(input_dim=input_dim)
        self.RQKern = RationalQuadraticKernel()
        self.PerKern = PERKernel()
        self.linear = nn.Linear(base_kernel_num, 1)

    def forward(self,x1,x2):
        """
        Compute the covariance matrix using the Matern kernel with scalar length scale.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The covariance matrix.

        """
        var1 = self.RBFKernel.forward(x1,x2)
        var2 = self.Linekern.forward(x1, x2).to_dense()
        var3 = self.RQKern.forward(x1, x2)
        var = torch.stack((var1, var2,var3), dim=-1)  # 沿着新的维度（第三维）合并
        var_lin = self.linear(var).reshape(var1.shape)
        var = torch.exp(var_lin)
        return var        
