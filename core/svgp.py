import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import core.GP_CommonCalculation as GP  # Assuming GP_CommonCalculation contains necessary utilities
from core.kernel import ARDKernel
import data_sample.generate_example_data as data

# Constants
JITTER = 1e-3
PI = 3.1415
torch.manual_seed(4)


class svgp(nn.Module):
    def __init__(self, num_inducing, input_dim,num_data):
        super(svgp, self).__init__()

        self.num_inducing = num_inducing
        self.kernel = ARDKernel(input_dim)
        self.num_data = num_data
        # Inducing points
        self.xm = nn.Parameter(torch.rand(self.num_inducing, input_dim, dtype=torch.float64))  # Inducing points
        self.qu_mean = nn.Parameter(torch.zeros(self.num_inducing, 1, dtype=torch.float64))
        self.chole = nn.Parameter(torch.rand(self.num_inducing, 1, dtype=torch.float64))

        # Gaussian noise
        self.log_beta = nn.Parameter(torch.ones(1, dtype=torch.float64) * 0)

    def loss_function(self, X, Y):
        K_mm = self.kernel(self.xm, self.xm) + JITTER * torch.eye(self.xm.size(0), dtype=torch.float64,
                                                                  device=self.xm.device)
        Lm = torch.linalg.cholesky(K_mm)
        K_mm_inv = torch.cholesky_inverse(Lm)
        K_mn = self.kernel(self.xm, X)
        K_nm = K_mn.t()
        qu_S = self.chole @ self.chole.t() + JITTER * torch.eye(self.xm.size(0),
                                                                dtype=torch.float64,
                                                                device=self.xm.device)  # Ensure positive definite
        Ls = torch.linalg.cholesky(qu_S)
        K_nn = self.kernel(X, X).diag()
        batch_size = X.size(0)
        # K_nm * K_mm_inv * m, (b, 1)
        mean_vector = K_nm @ K_mm_inv @ self.qu_mean

        # diag(K_tilde), (b, 1)
        precision = 1 / self.log_beta.exp()
        K_tilde = precision * (K_nn - (K_nm @ K_mm_inv @ K_mn).diag())
        # k_i \cdot k_i^T, (b, m, m)
        lambda_mat = K_mm_inv @ K_mn @ K_nm @ K_mm_inv
        # Trace terms, (b,)
        S_Ai = qu_S @ lambda_mat
        traces = precision * torch.trace(S_Ai)
        # Likelihood
        likelihood_sum = -0.5 * batch_size * torch.log(2 * torch.tensor(PI)) + 0.5 * batch_size * torch.log(
            self.log_beta.exp()) \
                         - 0.5 * self.log_beta.exp() * ((Y - mean_vector) ** 2).sum(dim=0).view(-1,
                                                                                                1) - 0.5 * torch.sum(
            K_tilde) - 0.5 * traces

        # Compute KL
        logdetS = 2 * Ls.diag().abs().log().sum()
        logdetKmm = 2 * Lm.diag().abs().log().sum()
        KL = 0.5 * (K_mm_inv @ qu_S).diag().sum(dim=0).view(-1, 1) + 0.5 * (self.qu_mean.t() @ K_mm_inv @ self.qu_mean) \
             - 0.5 * logdetS + 0.5 * logdetKmm - 0.5 * self.num_inducing
        variational_loss = KL - likelihood_sum * self.num_data / batch_size
        return variational_loss

    def forward(self, Xte):
        K_mm = self.kernel(self.xm, self.xm) + JITTER * torch.eye(self.xm.size(0), dtype=torch.float64,
                                                                  device=self.xm.device)
        Lm = torch.linalg.cholesky(K_mm)
        K_mm_inv = torch.cholesky_inverse(Lm)
        K_tt = self.kernel(Xte, Xte)
        K_tm = self.kernel(Xte, self.xm)
        A = K_tm @ K_mm_inv  # (t, m)
        mean = A @ self.qu_mean  # (t, 1)
        yvar = K_tt - K_tm @ K_mm_inv @ K_tm.t() + K_tm @ K_mm_inv @ (self.chole @ self.chole.t()) @ K_mm_inv @ K_tm.t()
        yvar = yvar.diag().view(-1, 1)
        return mean, yvar


if __name__ == '__main__':
    # Set random seed for reproducibility
    torch.manual_seed(4)

    # Train set
    num_data = 2000
    xtr, ytr, xte, yte = data.generate(num_data, 500, seed=2, input_dim=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xtr = xtr.to(device)
    ytr = ytr.to(device)
    xte = xte.to(device)
    yte = yte.to(device)

    # Perform normalization outside the model
    normalizer = GP.DataNormalization(method='standard')
    normalizer.fit(xtr, 'x')
    normalizer.fit(ytr, 'y')
    xtr_normalized = normalizer.normalize(xtr, 'x')
    ytr_normalized = normalizer.normalize(ytr, 'y')
    xte_normalized = normalizer.normalize(xte, 'x')

    # Create TensorDataset and DataLoader for minibatch training
    dataset = TensorDataset(xtr_normalized, ytr_normalized)
    batch_size = 500
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training the model
    num_inducing = 100
    learning_rate = 0.1
    num_epochs = 250  # Adjust as needed
    model = svgp(num_inducing=num_inducing, input_dim=xtr_normalized.size(1), num_data=num_data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    import time

    iteration_times = []
    num_data = xtr.size(0)
    # Training loop
    for epoch in range(num_epochs):
        for X_batch, Y_batch in dataloader:
            start_time = time.time()

            optimizer.zero_grad()
            loss = model.loss_function(X_batch, Y_batch)
            loss.backward()
            optimizer.step()

            end_time = time.time()
            iteration_times.append(end_time - start_time)

        print(f'Epoch {epoch}, Loss: {loss.item()}')

    average_iteration_time = sum(iteration_times) / len(iteration_times)
    print(f'Average iteration time: {average_iteration_time:.5f} seconds')

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        predictions, var = model(xte_normalized)
        predictions = normalizer.denormalize(predictions, 'y')
        var = normalizer.denormalize_cov(var, 'y')
        mse = torch.mean((predictions - yte) ** 2)
        print(f'Test MSE: {mse.item()}')
        plt.figure()
        plt.plot(xte.cpu().numpy(), yte.cpu().numpy(), 'r.')
        plt.plot(xte.cpu().numpy(), predictions.cpu().numpy(), 'b-')
        plt.fill_between(xte.cpu().numpy().reshape(-1),
                         (predictions - 1.96 * var.sqrt()).cpu().numpy().reshape(-1),
                         (predictions + 1.96 * var.sqrt()).cpu().numpy().reshape(-1), alpha=0.2)
        plt.show()
