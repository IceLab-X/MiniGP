
# Author: Wei W. Xing (wxing.me)
# Email: wayne.xingle@gmail.com
# Date: 2023-11-26
from data_sample import generate_example_data as data
import numpy as np
import torch
import torch.nn as nn
from core.kernel import ARDKernel
import core.GP_CommonCalculation as gp_pack

import matplotlib.pyplot as plt

EPS = 1e-10
class deepkernelGP(nn.Module):
    """
    A Gaussian Process model with a deep kernel learned through a neural network.

    Args:
        input_dim (int): Dimensionality of the input data.
        kernel (callable): The kernel function to use. Defaults to ARDKernel.
        log_beta (torch.Tensor, optional): Initial value for the log noise level. If None, initialized to 0.
        layer_structure (list, optional): List of integers defining the structure of the neural network layers.
                                           If None, a default structure is used.
    """
    def __init__(self, input_dim, kernel=ARDKernel, log_beta=None, layer_structure=None):
        super(deepkernelGP, self).__init__()

        # GP hyperparameters
        if log_beta is None:
            self.log_beta = nn.Parameter(torch.ones(1) * 0)  # Initialize log_beta to 0 if not provided
        else:
            self.log_beta = nn.Parameter(log_beta)

        # Define the layer structure of the neural network for feature extraction
        layers = []
        if layer_structure is None:
            layer_structure = [input_dim, input_dim * 10, input_dim *5, input_dim]

        for i in range(len(layer_structure) - 1):
            layers.append(nn.Linear(layer_structure[i], layer_structure[i + 1]))
            if i < len(layer_structure) - 2:  # Do not apply activation on the last layer
                layers.append(nn.LeakyReLU())

        self.FeatureExtractor = nn.Sequential(*layers)
        self.kernel = kernel(layer_structure[-1])

    def forward(self, x_train, y_train, x_test):
        """
        Compute the mean and covariance for the posterior distribution.

        Args:
            x_train (torch.Tensor): Training input data.
            y_train (torch.Tensor): Training target data.
            x_test (torch.Tensor): Test input data.

        Returns:
            mu (torch.Tensor): The predicted mean of the test data.
            cov (torch.Tensor): The predicted covariance of the test data.
        """
        x_train = self.FeatureExtractor(x_train)
        x_test = self.FeatureExtractor(x_test)

        K = self.kernel(x_train, x_train) + self.log_beta.exp().pow(-1) * torch.eye(x_train.shape[0])
        K_s = self.kernel(x_train, x_test)
        K_ss = self.kernel(x_test, x_test)

        mu, cov = gp_pack.conditional_Gaussian(y_train, K, K_s, K_ss)
        cov = cov.sum(dim=0).view(-1, 1) + self.log_beta.exp().pow(-1)

        return mu, cov

    def negative_log_likelihood(self, x_train, y_train):
        """
        Compute the negative log-likelihood of the Gaussian Process.

        Args:
            x_train (torch.Tensor): Training input data.
            y_train (torch.Tensor): Training target data.

        Returns:
            torch.Tensor: The negative log-likelihood value.
        """
        x_train = self.FeatureExtractor(x_train)
        K = self.kernel(x_train, x_train) + self.log_beta.exp().pow(-1) * torch.eye(x_train.shape[0])

        return -gp_pack.Gaussian_log_likelihood(y_train, K)

    def train_adam(self, x_train, y_train, niteration=10, lr=0.1):
        """
        Train the model using the Adam optimizer.

        Args:
            x_train (torch.Tensor): Training input data.
            y_train (torch.Tensor): Training target data.
            niteration (int, optional): Number of iterations for training. Defaults to 10.
            lr (float, optional): Learning rate. Defaults to 0.1.

        Returns:
            torch.Tensor: The final loss value after training.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for i in range(niteration):
            optimizer.zero_grad()
            loss = self.negative_log_likelihood(x_train, y_train)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('iter %d, loss %.3f' % (i, loss.item()))
        return loss

    def print_parameters(self):
        """
        Print the names and shapes of all parameters in the model.
        """
        for name, param in self.named_parameters():
            print(f"Parameter name: {name}, shape: {param.shape}")


if __name__ == "__main__":
    print('Testing deepkernelGP model')
    print(torch.__version__)

    torch.manual_seed(seed=2)

    # Generate synthetic test data
    xte = torch.rand(128, 2) * 2
    yte = torch.sin(xte.sum(1)).view(-1, 1) + 10

    # Generate synthetic training data
    xtr = torch.rand(232, 2) * 2
    ytr = torch.sin(xtr.sum(1)).view(-1, 1) + torch.randn(232, 1) * 0.1 + 10

    # Normalize the data
    import core.GP_CommonCalculation as GP
    normalizer = GP.DataNormalization()
    normalizer.fit(xtr, 'x')
    normalizer.fit(ytr, 'y')
    xtr_normalized = normalizer.normalize(xtr, 'x')
    ytr_normalized = normalizer.normalize(ytr, 'y')
    xte_normalized = normalizer.normalize(xte, 'x')

    # Define the layer structure for the neural network
    layer_structure = [2,20,10, 2]

    # Initialize and train the model
    model = deepkernelGP(input_dim=2, layer_structure=layer_structure)
    model.train_adam(xtr_normalized, ytr_normalized, 800, lr=0.03)

    with torch.no_grad():
        # Make predictions on the test data
        ypred, ypred_var = model.forward(xtr_normalized, ytr_normalized, xte_normalized)
        ypred = normalizer.denormalize(ypred, 'y')
        ypred_var = normalizer.denormalize_cov(ypred_var, 'y')

    # Plot the results
    plt.plot(xte.sum(1), yte, 'b+', label='True values')
    plt.plot(xte.sum(1), ypred.reshape(-1).detach(), 'r+', label='Predicted values')
    plt.legend()
    plt.show()

    # Scatter plot of true vs predicted values
    plt.scatter(yte, ypred.reshape(-1).detach())
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.show()
