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


# util functions to compute the log likelihood.
def conjugate_gradient(A, b, x0=None, tol=1e-1, max_iter=1000):
    """
       Solve a system of linear equations Ax = b (equivalently x=A^{-1} b) using the Conjugate Gradient method.
        tool function to compute the log likelihood
       Parameters:
       A (torch.Tensor): The square, symmetric, positive-definite matrix A.
       b (torch.Tensor): The right-hand side vector b.
       x0 (torch.Tensor, optional): The initial guess for the solution vector x. If None, defaults to a zero vector.
       tol (float, optional): The tolerance for the stopping criterion. Defaults to 1e-1.
       max_iter (int, optional): The maximum number of iterations. Defaults to 1000.

       Returns:
       torch.Tensor: The solution vector x.
    """
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0
    r = b - torch.matmul(A, x)
    p = r.clone()
    rsold = torch.dot(r.flatten(), r.flatten())

    for i in range(max_iter):
        Ap = torch.matmul(A, p)
        alpha = rsold / torch.dot(p.flatten(), Ap.flatten())
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r.flatten(), r.flatten())

        if torch.sqrt(rsnew) < tol:
            break

        p = r + (rsnew / rsold) * p
        rsold = rsnew

        # if i % 10 == 0:  # Print diagnostics every 10 iterations
        # print(f"Iteration {i}: Residual norm {torch.sqrt(rsnew):.6e}")

    return x


def compute_inverse_and_log_det_positive_eigen(matrix):
    """
    Perform eigen decomposition, remove non-positive eigenvalues,
    and compute the inverse matrix and the log determinant of the matrix.

    Parameters:
    matrix (torch.Tensor): Input matrix for decomposition.

    Returns:
    torch.Tensor: Inverse of the matrix with non-positive eigenvalues removed.
    torch.Tensor: Log determinant of the matrix with non-positive eigenvalues removed.
    """
    eigenvalues, eigenvectors = torch.linalg.eig(matrix)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    # print(eigenvalues)
    positive_indices = eigenvalues > 1e-4
    removed_count = torch.sum(~positive_indices).item()
    # if removed_count > 0:
    # print(f"Removed {removed_count} small or non-positive eigenvalue(s).")
    eigenvalues = eigenvalues[positive_indices]
    eigenvectors = eigenvectors[:, positive_indices]
    inv_eigenvalues = torch.diag(1.0 / eigenvalues)
    inverse_matrix = eigenvectors @ inv_eigenvalues @ eigenvectors.T
    log_det_K = torch.sum(torch.log(eigenvalues))
    return inverse_matrix, log_det_K


# compute the log likelihood of a normal distribution
def Gaussian_log_likelihood(y, cov, Kinv_method='cholesky'):
    """
    Compute the log-likelihood of a Gaussian distribution N(y|0, cov). If you have a mean mu, you can use N(y|mu, cov) = N(y-mu|0, cov).

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

    if Kinv_method == 'cholesky':
        # fastest implementation so far for any covariance matrix
        L = torch.linalg.cholesky(cov)
        # return -0.5 * (y_use.T @ torch.cholesky_solve(y_use, L) + L.diag().log().sum() + len(x_train) * np.log(2 * np.pi))
        if y.shape[1] > 1:
            Warning(
                'y_use.shape[1] > 1, will treat each column as a sample (for the joint normal distribution) and sum the log-likelihood')
            # 
            # (Alpha ** 2).sum() = (Alpha @ Alpha^T).diag().sum() = \sum_i (Alpha @ Alpha^T)_{ii}
            # 
            y_dim = y.shape[1]
            log_det_K = 2 * torch.sum(torch.log(torch.diag(L)))
            gamma = torch.linalg.solve_triangular(L, y, upper=False)
            return - 0.5 * ((gamma ** 2).sum() + log_det_K * y_dim + len(y) * y_dim * np.log(2 * np.pi))
        else:
            gamma = torch.linalg.solve_triangular(L, y, upper=False)
            return -0.5 * (gamma.T @ gamma + 2 * L.diag().log().sum() + len(y) * np.log(2 * np.pi))

    elif Kinv_method == 'torch_distribution_MN1':
        L = torch.linalg.cholesky(cov)
        return torch.distributions.MultivariateNormal(y, scale_tril=L).log_prob(y)
    elif Kinv_method == 'torch_distribution_MN2':
        return torch.distributions.MultivariateNormal(y, cov).log_prob(y)
    elif Kinv_method == 'eigen':
        K_inv, log_det_K = compute_inverse_and_log_det_positive_eigen(cov)
        return -0.5 * (y.T @ K_inv @ y + log_det_K + len(y) * np.log(2 * np.pi))
    elif Kinv_method == 'conjugate':
        L = torch.linalg.cholesky(cov)
        Sigma_inv_y = conjugate_gradient(cov, y)

        return -0.5 * (torch.matmul(y.t(), Sigma_inv_y) - 0.5 * len(y) * torch.log(
            2 * torch.tensor(torch.pi))) - L.diag().log().sum()
    else:
        raise ValueError('Kinv_method should be either direct or cholesky')


def conditional_Gaussian(y, Sigma, K_s, K_ss, Kinv_method='cholesky'):
    # Sigma = Sigma + torch.eye(len(Sigma)) * EPS

    if Kinv_method == 'cholesky':
        # recommended implementation, fastest so far
        L = torch.linalg.cholesky(Sigma)
        alpha = torch.cholesky_solve(y, L)
        mu = K_s.T @ alpha
        # v = torch.cholesky_solve(K_s, L)    # wrong implementation
        # v = L.inverse() @ K_s   # correct implementation
        v = torch.linalg.solve_triangular(L, K_s, upper=False)
        cov = K_ss - v.T @ v
    elif Kinv_method == 'conjugate':
        K_inv_y = conjugate_gradient(Sigma, y)
        mu = K_s.T @ K_inv_y
        K_inv_K_s = conjugate_gradient(Sigma, K_s)
        cov = K_ss - K_s.T @ K_inv_K_s
    else:
        raise ValueError('Kinv_method should be either direct or cholesky')

    return mu, cov


class DataNormalization(nn.Module):
    """
    A class used to perform data normalization for different datasets using standardization or min-max scaling.

    Parameters
    ----------
    method : str, optional
        The normalization method to use. Can be 'standard' for standardization or 'min_max' for min-max scaling (default is 'standard').
    mode : int, optional
        The mode of normalization. If 0, normalization is performed feature-wise. If 1, normalization is performed on the entire dataset (default is 0).
    learnable : bool, optional
        If True, the normalization parameters (mean, std, min, max) will be learnable parameters (default is False).
    eps : float, optional
        A small value to avoid division by zero (default is 1e-8).

    Methods
    -------
    fit(data, dataset_name)
        Computes and stores the normalization parameters for the given dataset.

    normalize(data, dataset_name)
        Normalizes the given data using the stored normalization parameters for the specified dataset.

    denormalize(normalized_data, dataset_name)
        Denormalizes the given data using the stored normalization parameters for the specified dataset.

    denormalize_cov(normalized_cov, dataset_name)
        Denormalizes the covariance matrix using the stored normalization parameters for the specified dataset.
    """

    def __init__(self, method="standard", mode=0, learnable=False, eps=1e-8):
        super(DataNormalization, self).__init__()
        self.method = method
        self.mode = mode
        self.learnable = learnable
        self.eps = eps
        self.params = {}

        if method not in ["standard", "min_max"]:
            raise ValueError("Method must be 'standard' or 'min_max'")

    def fit(self, data, dataset_name):
        device = data.device
        if self.mode == 0:
            if self.method == "standard":
                mean_vals = torch.mean(data, dim=0, keepdim=True).to(device)
                std_vals = torch.std(data, dim=0, keepdim=True).to(device)
                self.params[dataset_name] = {'mean': nn.Parameter(mean_vals) if self.learnable else mean_vals,
                                             'std': nn.Parameter(std_vals) if self.learnable else std_vals}
            elif self.method == "min_max":
                min_vals = torch.min(data, dim=0, keepdim=True).values.to(device)
                max_vals = torch.max(data, dim=0, keepdim=True).values.to(device)
                self.params[dataset_name] = {'min': nn.Parameter(min_vals) if self.learnable else min_vals,
                                             'max': nn.Parameter(max_vals) if self.learnable else max_vals}
        elif self.mode == 1:
            if self.method == "standard":
                mean_vals = torch.mean(data).to(device)
                std_vals = torch.std(data).to(device)
                self.params[dataset_name] = {'mean': nn.Parameter(mean_vals) if self.learnable else mean_vals,
                                             'std': nn.Parameter(std_vals) if self.learnable else std_vals}
            elif self.method == "min_max":
                min_vals = torch.min(data).to(device)
                max_vals = torch.max(data).to(device)
                self.params[dataset_name] = {'min': nn.Parameter(min_vals) if self.learnable else min_vals,
                                             'max': nn.Parameter(max_vals) if self.learnable else max_vals}

    def normalize(self, data, dataset_name):
        if dataset_name not in self.params:
            raise ValueError(f"No parameters found for dataset '{dataset_name}'. Please fit the data first.")

        device = data.device

        if self.method == "standard":
            mean_vals = self.params[dataset_name]['mean'].to(device)
            std_vals = self.params[dataset_name]['std'].to(device)
            return (data - mean_vals.expand_as(data)) / (std_vals.expand_as(data) + self.eps)
        elif self.method == "min_max":
            min_vals = self.params[dataset_name]['min'].to(device)
            max_vals = self.params[dataset_name]['max'].to(device)
            return (data - min_vals.expand_as(data)) / ((max_vals - min_vals).expand_as(data) + self.eps)

    def denormalize(self, normalized_data, dataset_name):
        if dataset_name not in self.params:
            raise ValueError(f"No parameters found for dataset '{dataset_name}'. Please fit the data first.")

        device = normalized_data.device

        if self.method == "standard":
            mean_vals = self.params[dataset_name]['mean'].to(device)
            std_vals = self.params[dataset_name]['std'].to(device)
            return normalized_data * (std_vals.expand_as(normalized_data) + self.eps) + mean_vals.expand_as(
                normalized_data)
        elif self.method == "min_max":
            min_vals = self.params[dataset_name]['min'].to(device)
            max_vals = self.params[dataset_name]['max'].to(device)
            return normalized_data * ((max_vals - min_vals).expand_as(normalized_data) + self.eps) + min_vals.expand_as(
                normalized_data)

    def denormalize_cov(self, normalized_cov, dataset_name):
        if dataset_name not in self.params:
            raise ValueError(f"No parameters found for dataset '{dataset_name}'. Please fit the data first.")

        device = normalized_cov.device

        if self.method == "standard":
            std_vals = self.params[dataset_name]['std'].to(device)
            std_vals = std_vals.view(-1, 1)  # Ensure std_vals is a column vector

            return normalized_cov * (std_vals.T  @ std_vals + self.eps)
        elif self.method == "min_max":
            min_vals = self.params[dataset_name]['min'].to(device)
            max_vals = self.params[dataset_name]['max'].to(device)
            range_vals = (max_vals - min_vals).view(-1, 1)  # Ensure range_vals is a column vector
            return normalized_cov * (range_vals.T @ range_vals + self.eps)


class Warp(nn.Module):
    """
        A class to apply and invert transformations using either the 'log' or 'kumar' method.

        Attributes:
            method (str): The method to use for transformations ('log' or 'kumar').
            a (torch.nn.Parameter): Parameter 'a' for the 'kumar' transformation, initialized as a torch tensor.
            b (torch.nn.Parameter): Parameter 'b' for the 'kumar' transformation, initialized as a torch tensor.
            a_lower_bound (float): Lower bound for parameter 'a'.
            a_upper_bound (float): Upper bound for parameter 'a'.
            b_lower_bound (float): Lower bound for parameter 'b'.
            b_upper_bound (float): Upper bound for parameter 'b'.
        """

    def __init__(self, method="kumar", initial_a=1.0, initial_b=1.0, warp_level=0.25):
        """
        Initialize the Warp class with the specified method and initial parameters.

        Args:
            method (str): The method to use for transformations ('log' or 'kumar'). Default is 'kumar'.
            initial_a (float): Initial value for parameter 'a'. Default is 1.0.
            initial_b (float): Initial value for parameter 'b'. Default is 1.0.
            warp_level (float): Warp level to determine bounds for parameters 'a' and 'b'. Default is 0.25.
        """
        super(Warp, self).__init__()
        self.method = method
        self.a = nn.Parameter(torch.tensor(initial_a, dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor(initial_b, dtype=torch.float64))
        # Define the parameter bounds
        self.a_lower_bound = 1.0 - min(1.0, warp_level)
        self.a_upper_bound = 1.0 + warp_level
        self.b_lower_bound = 1.0 - min(1.0, warp_level)
        self.b_upper_bound = 1.0 + warp_level

    def transform(self, y):
        """
        Apply the specified transformation to the input.

        Args:
            y (torch.Tensor): The input tensor to transform.

        Returns:
            torch.Tensor: The transformed tensor.
        """

        a = self._bounded_param(self.a, self.a_lower_bound, self.a_upper_bound)
        b = self._bounded_param(self.b, self.b_lower_bound, self.b_upper_bound)
        if self.method == 'log':
            return torch.log(y + 1)
        elif self.method == 'kumar':
            return 1 - (1 - y ** a) ** b
        else:
            return y

    def grad(self, y):
        """
        Compute the gradient of the transformation. This is used to compute the Jacobian of the transformation, which is required to compute the correct likelyhood.

        Args:
            y (torch.Tensor): The input tensor to compute the gradient for.

        Returns:
            torch.Tensor: The gradient tensor.

        Raises:
            NotImplementedError: If the gradient computation is not implemented for the specified method.
        """
        if self.method == 'kumar':
            a = abs(self.a)  # Use built-in abs function
            b = abs(self.b)
            return a * b * (y ** (a - 1)) * ((1 - y ** a) ** (b - 1))
        else:
            raise NotImplementedError(f"Gradient not implemented for method '{self.method}'.")

    def _bounded_param(self, param, lower_bound, upper_bound):
        """
        Bound the parameter within the specified range using a sigmoid function.

        Args:
            param (torch.Tensor): The parameter to bound.
            lower_bound (float): The lower bound for the parameter.
            upper_bound (float): The upper bound for the parameter.

        Returns:
            torch.Tensor: The bounded parameter.
        """
        param_sigmoid = torch.sigmoid(param)
        return lower_bound + (upper_bound - lower_bound) * param_sigmoid

    def back_transform(self, mean_transformed, var_diag_transformed=None):
        """
       Apply the inverse transformation and compute the back-transformed mean and variance.

       Args:
           mean_transformed (torch.Tensor): The transformed mean tensor.
           var_diag_transformed (torch.Tensor, optional): The transformed variance tensor. Default is None.

       Returns:
           tuple: The back-transformed mean and variance tensors.

       Raises:
           NotImplementedError: If the back transformation is not implemented for the specified method.
       """
        if self.method == 'kumar':
            a = self._bounded_param(self.a, self.a_lower_bound, self.a_upper_bound)
            b = self._bounded_param(self.b, self.b_lower_bound, self.b_upper_bound)

            def inverse(mean_transformed):
                return (1 - (1 - mean_transformed) ** (1 / b)) ** (1 / a)

            median = inverse(mean_transformed)
            if var_diag_transformed is not None:
                var_diag = (inverse(mean_transformed + 2 * torch.sqrt(var_diag_transformed)) -
                            inverse(mean_transformed - 2 * torch.sqrt(var_diag_transformed)))
                var_diag = var_diag ** 2 / 4
                return median, var_diag
            else:
                return median

        elif self.method == 'log':
            mean = torch.exp(mean_transformed + 0.5 * var_diag_transformed) - 1
            var_diag = (torch.exp(var_diag_transformed) - 1) * torch.exp(2 * mean_transformed + var_diag_transformed)
            return mean, var_diag

        else:
            if var_diag_transformed is not None:
                return mean_transformed, var_diag_transformed
            else:
                raise NotImplementedError(f"Transform and inverse not implemented for method '{self.method}'.")


class XYdata_normalization:
    def __init__(self, X, Y=None, normal_y_mode=0):
        # Compute mean and standard deviation for X
        self.X_mean = X.mean(0)
        self.X_std = (X.std(0) + EPS)  # Avoid division by zero

        # Compute mean and standard deviation for Y if provided
        if Y is not None:
            if normal_y_mode == 0:
                self.Y_mean = Y.mean()
                self.Y_std = (Y.std() + EPS)
            else:
                self.Y_mean = Y.mean(0)
                self.Y_std = (Y.std(0) + EPS)

    def normalize(self, X, Y=None):
        # Normalize X
        X_normalized = (X - self.X_mean.expand_as(X)) / self.X_std.expand_as(X)
        # Normalize Y if provided
        if Y is not None:
            Y_normalized = (Y - self.Y_mean.expand_as(Y)) / self.Y_std.expand_as(Y)
            return X_normalized, Y_normalized
        return X_normalized

    def denormalize(self, X, Y=None):
        # Denormalize X
        X_denormalized = X * self.X_std.expand_as(X) + self.X_mean.expand_as(X)
        # Denormalize Y if provided
        if Y is not None:
            Y_denormalized = Y * self.Y_std.expand_as(Y) + self.Y_mean.expand_as(Y)
            return X_denormalized, Y_denormalized
        return X_denormalized

    def denormalize_y(self, mean, var_diag=None):
        # Denormalize the mean and variance of the prediction
        mean_denormalized = mean * self.Y_std.expand_as(mean) + self.Y_mean.expand_as(mean)
        if var_diag is not None:
            var_diag_denormalized = var_diag.expand_as(mean) * (self.Y_std ** 2)
            return mean_denormalized, var_diag_denormalized
        return mean_denormalized
