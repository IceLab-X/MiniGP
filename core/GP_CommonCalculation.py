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

        #if i % 10 == 0:  # Print diagnostics every 10 iterations
            #print(f"Iteration {i}: Residual norm {torch.sqrt(rsnew):.6e}")

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
    #if removed_count > 0:
        #print(f"Removed {removed_count} small or non-positive eigenvalue(s).")
    eigenvalues = eigenvalues[positive_indices]
    eigenvectors = eigenvectors[:, positive_indices]
    inv_eigenvalues = torch.diag(1.0 / eigenvalues)
    inverse_matrix = eigenvectors @ inv_eigenvalues @ eigenvectors.T
    log_det_K = torch.sum(torch.log(eigenvalues))
    return inverse_matrix, log_det_K

# compute the log likelihood of a normal distribution
def Gaussian_log_likelihood(y, cov, Kinv_method='cholesky3'):
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
    
    if Kinv_method == 'cholesky1':
        L = torch.linalg.cholesky(cov)
        L_inv = torch.inverse(L)
        K_inv = L_inv.T @ L_inv
        return -0.5 * (y.T @ K_inv @ y + torch.logdet(cov) + len(y) * np.log(2 * np.pi))
    elif Kinv_method == 'cholesky2':
        L = torch.linalgcholesky(cov)
        return -0.5 * (y.T @ torch.cholesky_solve(y, L) + torch.logdet(cov) + len(y) * np.log(2 * np.pi))
    
    elif Kinv_method == 'cholesky3':
        # fastest implementation so far for any covariance matrix
        L = torch.linalg.cholesky(cov)
        # return -0.5 * (y_use.T @ torch.cholesky_solve(y_use, L) + L.diag().log().sum() + len(x_train) * np.log(2 * np.pi))
        if y.shape[1] > 1:
            Warning('y_use.shape[1] > 1, will treat each column as a sample (for the joint normal distribution) and sum the log-likelihood')
            # 
            # (Alpha ** 2).sum() = (Alpha @ Alpha^T).diag().sum() = \sum_i (Alpha @ Alpha^T)_{ii}
            # 
            y_dim = y.shape[1]
            log_det_K = 2 * torch.sum(torch.log(torch.diag(L)))
            gamma = torch.cholesky_solve(y, L, upper = False)
            return - 0.5 * ( (gamma ** 2).sum() + log_det_K * y_dim + len(y) * y_dim * np.log(2 * np.pi) )
        else:
            gamma = torch.linalg.solve_triangular(L,y,upper=False)
            return -0.5 * (gamma.T @ gamma + 2*L.diag().log().sum() + len(y) * np.log(2 * np.pi))

    elif Kinv_method == 'direct':
        # very slow
        K_inv = torch.inverse(cov)
        return -0.5 * (y.T @ K_inv @ y + torch.logdet(cov) + len(y) * np.log(2 * np.pi))
    elif Kinv_method == 'torch_distribution_MN1':
        L = torch.linalg.cholesky(cov)
        return torch.distributions.MultivariateNormal(y, scale_tril=L).log_prob(y)
    elif Kinv_method == 'torch_distribution_MN2':
        return torch.distributions.MultivariateNormal(y, cov).log_prob(y)
    elif Kinv_method == 'eigen':
        K_inv, log_det_K = compute_inverse_and_log_det_positive_eigen(cov)
        return -0.5 * (y.T @ K_inv @ y + log_det_K + len(y) * np.log(2 * np.pi))
    elif Kinv_method == 'conjugate':
        L= torch.linalg.cholesky(cov)
        Sigma_inv_y = conjugate_gradient(cov,y)

        return -0.5 * (torch.matmul(y.t(), Sigma_inv_y) - 0.5*len(y) * torch.log(2 * torch.tensor(torch.pi))) -L.diag().log().sum()
    else:
        raise ValueError('Kinv_method should be either direct or cholesky')
    
def conditional_Gaussian(y, Sigma, K_s, K_ss, Kinv_method='cholesky3'):
    # Sigma = Sigma + torch.eye(len(Sigma)) * EPS
    if Kinv_method == 'cholesky1':   # kernel inverse is not stable, use cholesky decomposition instead
        L = torch.linalg.cholesky(Sigma)
        L_inv = torch.inverse(L)
        K_inv = L_inv.T @ L_inv
        alpha = K_inv @ y
        mu = K_s.T @ alpha
        v = L_inv @ K_s
        cov = K_ss - v.T @ v
    elif Kinv_method == 'cholesky3':
        # recommended implementation, fastest so far
        L = torch.linalg.cholesky(Sigma)
        alpha = torch.cholesky_solve(y, L)
        mu = K_s.T @ alpha
        # v = torch.cholesky_solve(K_s, L)    # wrong implementation
        #v = L.inverse() @ K_s   # correct implementation
        v= torch.linalg.solve_triangular(L,K_s,upper=False)
        cov = K_ss - v.T @ v
    elif Kinv_method == 'direct':
        K_inv = torch.inverse(Sigma)
        mu = K_s.T @ K_inv @ y
        cov = K_ss - K_s.T @ K_inv @ K_s
    elif Kinv_method == 'conjugate':
        K_inv_y= conjugate_gradient(Sigma,y)
        mu = K_s.T @ K_inv_y
        K_inv_K_s = conjugate_gradient(Sigma,K_s)
        cov = K_ss - K_s.T @ K_inv_K_s
    else:
        raise ValueError('Kinv_method should be either direct or cholesky')
    
    return mu, cov

# class data_normalization:
#     def __init__(self, X, normal_mode=0):
#         # Compute mean and standard deviation for X
#         self.X_mean = X.mean(normal_mode)
#         self.X_std = (X.std(normal_mode) + EPS) # Avoid division by zero
class DataNormalization(nn.Module):
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
        if self.mode == 0:
            if self.method == "standard":
                mean_vals = torch.mean(data, dim=0, keepdim=True)
                std_vals = torch.std(data, dim=0, keepdim=True)
                self.params[dataset_name] = {'mean': nn.Parameter(mean_vals) if self.learnable else mean_vals,
                                             'std': nn.Parameter(std_vals) if self.learnable else std_vals}
            elif self.method == "min_max":
                min_vals = torch.min(data, dim=0, keepdim=True).values
                max_vals = torch.max(data, dim=0, keepdim=True).values
                self.params[dataset_name] = {'min': nn.Parameter(min_vals) if self.learnable else min_vals,
                                             'max': nn.Parameter(max_vals) if self.learnable else max_vals}
        elif self.mode == 1:
            if self.method == "standard":
                mean_vals = torch.mean(data)
                std_vals = torch.std(data)
                self.params[dataset_name] = {'mean': nn.Parameter(mean_vals) if self.learnable else mean_vals,
                                             'std': nn.Parameter(std_vals) if self.learnable else std_vals}
            elif self.method == "min_max":
                min_vals = torch.min(data)
                max_vals = torch.max(data)
                self.params[dataset_name] = {'min': nn.Parameter(min_vals) if self.learnable else min_vals,
                                             'max': nn.Parameter(max_vals) if self.learnable else max_vals}

    def normalize(self, data, dataset_name):
        if dataset_name not in self.params:
            raise ValueError(f"No parameters found for dataset '{dataset_name}'. Please fit the data first.")

        if self.method == "standard":
            mean_vals = self.params[dataset_name]['mean']
            std_vals = self.params[dataset_name]['std']
            return (data - mean_vals.expand_as(data)) / (std_vals.expand_as(data) + self.eps)
        elif self.method == "min_max":
            min_vals = self.params[dataset_name]['min']
            max_vals = self.params[dataset_name]['max']
            return (data - min_vals.expand_as(data)) / ((max_vals - min_vals).expand_as(data) + self.eps)

    def denormalize(self, normalized_data, dataset_name):
        if dataset_name not in self.params:
            raise ValueError(f"No parameters found for dataset '{dataset_name}'. Please fit the data first.")

        if self.method == "standard":
            mean_vals = self.params[dataset_name]['mean']
            std_vals = self.params[dataset_name]['std']
            return normalized_data * (std_vals.expand_as(normalized_data) + self.eps) + mean_vals.expand_as(
                normalized_data)
        elif self.method == "min_max":
            min_vals = self.params[dataset_name]['min']
            max_vals = self.params[dataset_name]['max']
            return normalized_data * ((max_vals - min_vals).expand_as(normalized_data) + self.eps) + min_vals.expand_as(
                normalized_data)
    def denormalize_cov(self, normalized_cov, dataset_name):
        if dataset_name not in self.params:
            raise ValueError(f"No parameters found for dataset '{dataset_name}'. Please fit the data first.")
        if self.method == "standard":
            std_vals = self.params[dataset_name]['std']
            return normalized_cov * (std_vals.T @ std_vals + self.eps)
        elif self.method == "min_max":
            min_vals = self.params[dataset_name]['min']
            max_vals = self.params[dataset_name]['max']
            return normalized_cov * ((max_vals - min_vals).T @ (max_vals - min_vals) + self.eps)


class Warp(nn.Module):
    """
    A PyTorch module for applying different warping transformations to input data.

    Supported transformations:
    - Logarithmic (log)
    - Hyperbolic tangent (tanh)
    - Kumaraswamy CDF (kumar)

    Attributes:
    method (str): The transformation method to use ('log', 'tanh', 'kumar').
    a (nn.Parameter): Learnable parameter used in 'tanh' and 'kumar' transformations.
    b (nn.Parameter): Learnable parameter used in 'tanh' and 'kumar' transformations.
    c (nn.Parameter): Learnable parameter used in 'tanh' transformation.

    Methods:
    transform(y): Applies the chosen transformation to the input tensor y.
    grad(y): Computes the gradient of the chosen transformation with respect to y.
    tanh_inv(mean_transformed, tol=1e-5, max_iter=1000): Computes the inverse of the 'tanh' transformation.
    kumar_inv(mean_transformed, tol=1e-5, max_iter=1000): Computes the inverse of the 'Kum' transformation.
    back_transform(mean_transformed, var_diag_transformed): Applies the inverse transformation to mean and variance.
    """

    def __init__(self, method="log", initial_a=0.0, initial_b=0.0):
        """
        Initializes the Warp module with the specified transformation method.

        Args:
        method (str): The transformation method to use ('log', 'tanh', 'Kum').
        """
        super(Warp, self).__init__()
        self.method = method
        self.a = nn.Parameter(torch.tensor(initial_a))
        self.b = nn.Parameter(torch.tensor(initial_b))


    def transform(self, y):
        """
        Applies the chosen transformation to the input tensor y.

        Args:
        y (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The transformed tensor.
        """
        if self.method == 'log':
            return torch.log(y + 1)  # Add 1 to avoid log(0) issues
        elif self.method == 'tanh':
            c = nn.Parameter(torch.zeros(1))
            f_y = self.a.exp() * torch.tanh(self.b.exp() * y + c) + y
            return f_y
        elif self.method == 'kumar':
            return 1 - (1 - y ** self.a.exp()) ** self.b.exp()

    def grad(self, y):
        """
        Computes the gradient of the chosen transformation with respect to y.

        Args:
        y (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The gradient tensor.
        """
        if self.method == 'tanh':
            df_y = self.b.exp() * (1 - torch.tanh(self.b.exp() * y + self.c) ** 2) + 1
            return df_y
        elif self.method == 'kumar':
            a = self.a.exp()
            b = self.b.exp()
            df_y = a * b * (y ** (a - 1)) * ((1 - y ** a) ** (b - 1))
            return df_y
        else:
            raise NotImplementedError(f"Gradient not implemented for method {self.method}")

    def inv_transform(self, mean_transformed, tol=1e-5, max_iter=1000):
        """
        Computes the inverse of the transformation using the Newton-Raphson method for 'tanh' and analytically for 'kumar'.

        Args:
        mean_transformed (torch.Tensor): The transformed mean tensor.
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations.

        Returns:
        torch.Tensor: The inverse-transformed mean tensor.
        """
        if self.method == 'tanh':
            initial_guess = mean_transformed.clone()  # Ensure we don't modify the input tensor directly
            for _ in range(max_iter):
                f_y = self.transform(initial_guess)
                df_y = self.grad(initial_guess)
                next_guess = initial_guess - (f_y - mean_transformed) / df_y

                if torch.abs(f_y - mean_transformed).max() < tol:
                    break
                initial_guess = next_guess
            mean = initial_guess
            return mean
        elif self.method == 'kumar':
            a = self.a.exp()
            b = self.b.exp()
            return (1 - (1 - mean_transformed) ** (1 / b)) ** (1 / a)

    def back_transform(self, mean_transformed, var_diag_transformed, tol=1e-5, max_iter=1000):
        """
        Applies the inverse transformation to mean and variance.

        Args:
        mean_transformed (torch.Tensor): The transformed mean tensor.
        var_diag_transformed (torch.Tensor): The transformed variance tensor.
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations.

        Returns:
        tuple: The inverse-transformed mean and variance tensors.
        """
        if self.method == 'log':
            mean = torch.exp(mean_transformed + 0.5 * var_diag_transformed) - 1
            var_diag = (torch.exp(var_diag_transformed) - 1) * torch.exp(2 * mean_transformed + var_diag_transformed)
            return mean, var_diag

        elif self.method == 'tanh':
            median = self.inv_transform(mean_transformed)
            var_diag = self.inv_transform(mean_transformed + 2 * torch.sqrt(var_diag_transformed)) - self.inv_transform(
                mean_transformed - 2 * torch.sqrt(var_diag_transformed))
            var_diag = var_diag ** 2 / 4  # As we used 2σ, we need to divide by 4
            return median, var_diag

        elif self.method == 'kumar':
            median = self.inv_transform(mean_transformed)
            var_diag = self.inv_transform(mean_transformed + 2 * torch.sqrt(var_diag_transformed)) - self.inv_transform(
                mean_transformed - 2 * torch.sqrt(var_diag_transformed))
            var_diag = var_diag ** 2 / 4
            return median, var_diag
class XYdata_normalization:
    def __init__(self, X, Y=None, normal_y_mode=0):
        # Compute mean and standard deviation for X
        self.X_mean = X.mean(0)
        self.X_std = (X.std(0) + EPS) # Avoid division by zero

        # Compute mean and standard deviation for Y if provided
        if Y is not None:
            if normal_y_mode == 0:
                self.Y_mean = Y.mean()
                self.Y_std = (Y.std() + EPS)
            else:
                self.Y_mean =Y.mean(0)
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
