#author: Zidong Chen
#date: 2024-06-22
#version: v1.0
#description: This script provides functions to fix non-positive definite matrices. Detailed demonstration can be found in the non_positive_definite_fixer_implement.ipynb in model_FAQ file.
import torch
import numpy as np

def remove_similar_data(x, y, threshold=1e-4):
    # Calculate pairwise distances
    distances = torch.cdist(x, x, p=2)

    # Create a mask to identify points within the threshold
    mask = distances < threshold

    # Zero out the diagonal to ignore self-comparison
    mask.fill_diagonal_(False)

    # Use the upper triangle of the mask to avoid double marking
    mask_upper = torch.triu(mask, diagonal=1)

    # Identify indices of points to keep
    to_remove = mask_upper.any(dim=0)
    to_keep = ~to_remove

    return x[to_keep], y[to_keep]


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

#eigendecomposition 
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
            Warning(
                'y_use.shape[1] > 1, will treat each column as a sample (for the joint normal distribution) and sum the log-likelihood')
            #
            # (Alpha ** 2).sum() = (Alpha @ Alpha^T).diag().sum() = \sum_i (Alpha @ Alpha^T)_{ii}
            #
            y_dim = y.shape[1]
            log_det_K = 2 * torch.sum(torch.log(torch.diag(L)))
            gamma = torch.cholesky_solve(y, L, upper=False)
            return - 0.5 * ((gamma ** 2).sum() + log_det_K * y_dim + len(y) * y_dim * np.log(2 * np.pi))
        else:
            gamma = torch.cholesky_solve(y, L)
            return -0.5 * (gamma.T @ gamma + L.diag().log().sum() + len(y) * np.log(2 * np.pi))

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
    else:
        raise ValueError('Kinv_method should be either direct, cholesky or eigen')



def train_adam_with_reset(model, niteration=100, lr=0.01, max_norm=5):
    """
    Train a model using the Adam optimizer with periodic parameter resets.

    This function trains a given model using the Adam optimizer. It periodically
    checks the norms of specified parameters and resets them to their initial
    values if their norms exceed a given maximum norm.

    Args:
        model (torch.nn.Module): The model to be trained. The model should have
            attributes `log_length_scale`, `log_scale`, and `log_beta`.
        niteration (int, optional): The number of training iterations. Defaults to 100.
        lr (float, optional): The learning rate for the Adam optimizer. Defaults to 0.01.
        max_norm (float, optional): The maximum allowed norm for the parameters. If the norm of
            any parameter exceeds this value, the parameters are reset. Defaults to 5.

    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # modify this if you have different parameter names
    initial_log_length_scale = model.kernel.length_scales.clone().detach()
    initial_log_scale = model.kernel.signal_variance.clone().detach()

    def reset_parameters():
        # Reset parameters to their initial state around initial values
        model.kernel.length_scales.data.uniform_(initial_log_length_scale - 0.1, initial_log_length_scale + 0.1)
        model.kernel.signal_variance.uniform_(initial_log_scale - 0.1, initial_log_scale + 0.1)

    for iteration in range(niteration):
        try:
            optimizer.zero_grad()
            loss = model.negative_lower_bound()

            loss.backward()  # Compute gradients

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # Clip gradients to avoid explosion

            optimizer.step()  # Update parameters

            # Check parameter norms and reset if necessary
            with torch.no_grad():
                for param_name, param in [('log_length_scale', model.kernel.length_scales), ('signal_variance', model.kernel.signal_variance)]:
                    param_norm = torch.norm(param)
                    if param_norm > max_norm:
                        print(f"Parameter {param_name} norm {param_norm} exceeds max_norm, resetting parameters.")
                        reset_parameters()
                        break
            if (iteration+1) % 10 == 0:  # Print the loss every 10 iterations
                print(f'Iteration {iteration+1}: Loss: {loss.item():.5f}')


        except Exception as e:  # Check if the non-positive definite issue is not caused by parameter explosion
            print(f'Error during optimization at iteration {iteration}: {e}')
            # Log specific parameters during the error
            print(f'Iteration {iteration} error state:')
            print(model.parameters())
            raise e
