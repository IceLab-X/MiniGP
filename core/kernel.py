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
Pi=3.1415926
torch.set_default_dtype(torch.float64)
class NeuralKernel(nn.Module):
    def __init__(self, input_dim):
        super(NeuralKernel, self).__init__()

        # 定义核函数
        self.kernels = nn.ModuleDict({
            'RationalQuadratic': RationalQuadraticKernel(input_dim),
            'linear': LinearKernel(input_dim),
            'periodic': PeriodicKernel(input_dim),
            'ARD':ARDKernel(input_dim)
        })
        self.softplus= nn.Softplus()
        # 定义核函数的可学习权重
        self.weights = nn.ParameterDict({
            'RationalQuadratic': nn.Parameter(torch.tensor(1.0)),
            'linear': nn.Parameter(torch.tensor(1.0)),
            'periodic': nn.Parameter(torch.tensor(1.0)),
            'ARD':nn.Parameter(torch.tensor(1.0))
        })

    def forward(self, x1, x2):
        # 计算每个核函数的输出并加权相加
        K_sum = sum(self.softplus(self.weights[name]) * kernel(x1, x2)
                    for name, kernel in self.kernels.items())
        return K_sum
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
        initial_log_length_scale (float): The initial length scale value. Default is 0.
        initial_log_signal_variance (float): The initial signal variance value. Default is 0.
        input_dtype (torch.dtype): The data type of the input. Default is torch.float64.

    Attributes:
        log_length_scale (nn.Parameter): The length scales for each dimension.
        log_signal_variance  (nn.Parameter): The signal variance.
        eps (float): A small constant to prevent division by zero.

    """

    def __init__(self, input_dim, initial_log_length_scale=0.0, initial_log_signal_variance=0.0, input_dtype=torch.float64):
        super().__init__()

        self.length_scales = nn.Parameter(torch.ones(input_dim, dtype=input_dtype) * initial_log_length_scale)
        self.signal_variance = nn.Parameter(torch.tensor([initial_log_signal_variance]))


    def forward(self, x1, x2):
        """
        Compute the covariance matrix using the ARD kernel.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The covariance matrix.

        """
        device = x1.device
        length_scales = torch.exp(self.length_scales).to(device)
        signal_variance=self.signal_variance.exp().to(device)
        scaled_x1 = x1 / length_scales
        scaled_x2 = x2 / length_scales
        sqdist = torch.cdist(scaled_x1, scaled_x2, p=2)**2
        return signal_variance.exp() * torch.exp(-0.5 * sqdist)


    
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

    def __init__(self, input_dim,length_scales=1.0, signal_variance=1.0, alpha=1.0,eps=EPS):
        super(RationalQuadraticKernel, self).__init__()
        self.length_scales = nn.Parameter(torch.ones(input_dim) * length_scales)
        self.signal_variance = nn.Parameter(torch.tensor([signal_variance]))
        self.alpha = nn.Parameter(torch.tensor([alpha]))
        self.eps= eps
    def forward(self, x1, x2):
        """
        Compute the covariance matrix using the rational quadratic kernel.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The covariance matrix.

        """
        length_scales = torch.abs(self.length_scales) + self.eps
        alpha=torch.abs(self.alpha) + self.eps

        scaled_x1 = x1 / length_scales
        scaled_x2 = x2 / length_scales
        sqdist = torch.cdist(scaled_x1, scaled_x2, p=2)**2
        return self.signal_variance.pow(2) * torch.pow(1 + 0.5 * sqdist / alpha , -alpha)
  
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

        
class RBFKernel(nn.Module):
    """
    Radial Basis Function (RBF) kernel module.

    Args:
        input_dim (int): The input dimension.
        initial_length_scale (float): The length scale value. Default is 0.0 (log space).

    Attributes:
        length_scales (nn.Parameter): The length scales for each input dimension.
    """
    def __init__(self, input_dim, initial_length_scale=0.0):
        super(RBFKernel, self).__init__()
        self.length_scales = nn.Parameter(torch.ones(input_dim) * initial_length_scale)

    def forward(self, x1, x2):
        """
        Computes the RBF kernel between inputs x1 and x2.

        Args:
            x1 (Tensor): Input tensor of shape (n, d).
            x2 (Tensor): Input tensor of shape (m, d).

        Returns:
            Tensor: The RBF kernel matrix of shape (n, m).
        """
        # Ensure length scales are positive
        length_scales = torch.exp(self.length_scales)

        # Scale inputs by length scales
        scaled_x1 = x1 / length_scales
        scaled_x2 = x2 / length_scales

        # Calculate the squared Euclidean distance
        sqdist = torch.cdist(scaled_x1, scaled_x2, p=2) ** 2

        # Compute the RBF kernel
        return torch.exp(-0.5 * sqdist)




class PeriodicKernel(nn.Module):
    """
        Periodic kernel module.

        Args:
            input_dim (int): The input dimension.
            initial_length_scale (float): The initial length scale value (log space). Default is 0.0.
            initial_period (float): The initial period value (log space). Default is 0.0.

        Attributes:
            length_scales (nn.Parameter): The length scales for each input dimension.
            period (nn.Parameter): The period.
        """

    def __init__(self, input_dim, initial_length_scale=1.0, initial_period=2.0):
        super(PeriodicKernel, self).__init__()
        self.length_scales = nn.Parameter(torch.ones(1) * initial_length_scale)
        self.period = nn.Parameter(torch.tensor(initial_period))

    def forward(self, x1, x2):
        """
        Computes the Periodic kernel between inputs x1 and x2.

        Args:
            x1 (Tensor): Input tensor of shape (n, d).
            x2 (Tensor): Input tensor of shape (m, d).

        Returns:
            Tensor: The Periodic kernel matrix of shape (n, m).
        """
        # Ensure length scales and period are positive
        length_scales = torch.abs(self.length_scales)+EPS
        period = torch.abs(self.period)+EPS

        # Scale inputs by length scales
        scaled_x1 = torch.tensor(Pi)*x1/period
        scaled_x2 = torch.tensor(Pi)*x2/period

        # Calculate the pairwise Euclidean distance
        dist = torch.cdist(scaled_x1, scaled_x2, p=2)

        # Compute the periodic kernel
        return torch.exp(-2 * (torch.sin(dist).pow(2)) / length_scales**2)
# class PeriodicKernel(nn.Module):
#     """
#     Periodic kernel module.
#
#     Args:
#         input_dim (int): The input dimension.
#         initial_length_scale (float): The initial length scale value (log space). Default is 0.0.
#         initial_period (float): The initial period value (log space). Default is 0.0.
#
#     Attributes:
#         length_scales (nn.Parameter): The length scales for each input dimension.
#         period (nn.Parameter): The period.
#     """
#
#     def __init__(self, input_dim, initial_length_scale=1.0, initial_period=1.0):
#         super(PeriodicKernel, self).__init__()
#         self.length_scales = nn.Parameter(torch.ones() * initial_length_scale)
#         self.period = nn.Parameter(torch.tensor(initial_period))
#
#     def forward(self, x1, x2):
#         """
#         Computes the Periodic kernel between inputs x1 and x2.
#
#         Args:
#             x1 (Tensor): Input tensor of shape (n, d).
#             x2 (Tensor): Input tensor of shape (m, d).
#
#         Returns:
#             Tensor: The Periodic kernel matrix of shape (n, m).
#         """
#         # Ensure length scales and period are positive
#         length_scales = torch.exp(self.length_scales) + EPS
#         period = torch.exp(self.period) + EPS
#
#         # Calculate the pairwise Euclidean distance
#         dist = torch.cdist(x1, x2, p=2)
#
#         # Compute the periodic kernel
#         scaled_dist = torch.tensor(Pi) * dist / period
#         periodic_kernel = torch.exp(-2 * (torch.sin(scaled_dist).pow(2)) / (length_scales ** 2))
#         print(1)
#         return periodic_kernel