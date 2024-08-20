import os
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from core.kernel import ARDKernel
import core.GP_CommonCalculation as GP
import data_sample.generate_example_data as data
from core.GP_CommonCalculation import conjugate_gradient as cg
from torch.utils.data import TensorDataset, DataLoader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Fixing strange error if run in MacOS

# Constants
JITTER = 1e-3
PI = 3.1415
torch.manual_seed(4)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)


class parametricGP(nn.Module):
    def __init__(self, kernel, input_dim, num_inducing, device='cpu'):
        super(parametricGP, self).__init__()

        self.kernel = kernel
        self.num_inducing = num_inducing
        self.jitter = JITTER
        self.device = device

        # Inducing points
        self.xm = nn.Parameter(torch.rand(num_inducing, input_dim, dtype=torch.float64).to(device))
        self.qu_mean = nn.Parameter(torch.rand(num_inducing, 1, dtype=torch.float64).to(device))

        # Gaussian noise
        self.log_beta = nn.Parameter(torch.ones(1, dtype=torch.float64).to(device) * -4)

    def predict(self, xtr):
        K_mm = self.kernel(self.xm, self.xm) + self.jitter * torch.eye(self.xm.size(0), dtype=torch.float64).to(
            self.device)
        K_tm = self.kernel(xtr, self.xm)
        ytr_pred = K_tm @ torch.linalg.solve(K_mm, self.qu_mean)
        return ytr_pred

    def forward(self, xte):
        K_mm = self.kernel(self.xm, self.xm) + self.jitter * torch.eye(self.xm.size(0), dtype=torch.float64).to(
            self.device)
        K_tm = self.kernel(xte, self.xm)
        K_tt_diag = self.kernel(xte, xte).diag()

        y_pred = K_tm @ torch.linalg.solve(K_mm, self.qu_mean)
        y_var = K_tt_diag - (K_tm @ torch.linalg.solve(K_mm, K_tm.t())).diag() + self.log_beta.exp().pow(-1)

        y_var = y_var.view(-1, 1)
        return y_pred, y_var

    def loss_function(self, xtr, ytr):
        ytr_pred = self.predict(xtr)
        n = ytr.size(0)
        loss = 0.5 * n * (
                torch.log(self.log_beta.exp().pow(-1)) + torch.log(2 * torch.tensor(PI, dtype=torch.float64))) + \
               0.5 * (ytr - ytr_pred).pow(2).sum() / self.log_beta.exp().pow(-1)
        return loss


if __name__ == '__main__':
    # Generate data
    xtr, ytr, xte, yte = data.generate(2000, 500, seed=2)
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

    # Training the model
    num_inducing = 100
    learning_rate = 0.1
    num_epochs = 800
    batchsize = 500

    # Create an instance of ParametricGP
    model = parametricGP(kernel=ARDKernel(1), input_dim=1, num_inducing=num_inducing, device=device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_values = []
    iteration_times = []

    # Prepare data loader for batching
    dataset = TensorDataset(xtr_normalized, ytr_normalized)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False)

    import time

    for i in range(num_epochs):
        start_time = time.time()

        for X_batch, Y_batch in dataloader:
            optimizer.zero_grad()
            loss = model.loss_function(X_batch, Y_batch)
            loss.backward()
            optimizer.step()

        end_time = time.time()

        loss_values.append(loss.item())
        iteration_times.append(end_time - start_time)

        if i % 10 == 0:
            print('iter', i, 'nll:{:.5f}'.format(loss.item()))

    average_iteration_time = sum(iteration_times) / len(iteration_times)
    print(f'Average iteration time: {average_iteration_time:.5f} seconds')

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        ypred_normalized, yvar_normalized = model.forward(xte_normalized)
        ypred = normalizer.denormalize(ypred_normalized, 'y')
        yvar = normalizer.denormalize_cov(yvar_normalized, 'y')
        mse = torch.mean((ypred - yte) ** 2)
        print(f'Test MSE: {mse.item()}')

    # Plotting the results
    plt.figure(figsize=(10, 6))
    xm = normalizer.denormalize(model.xm.cpu().detach(), 'x')
    qu_mean = normalizer.denormalize(model.qu_mean.detach(), 'y')
    plt.scatter(xm.cpu(), qu_mean.cpu(), color='red', label='Inducing points')
    plt.plot(xte.cpu().numpy(), yte.cpu().numpy(), 'b', label='True function')
    plt.plot(xte.cpu().numpy(), ypred.cpu().numpy(), 'r', label='Predicted function')
    plt.fill_between(xte.cpu().numpy().squeeze(),
                     (ypred.cpu().numpy() - 1.96 * torch.sqrt(yvar).cpu().numpy()).squeeze(),
                     (ypred.cpu().numpy() + 1.96 * torch.sqrt(yvar).cpu().numpy()).squeeze(), alpha=0.2)
    plt.xlim(0, 1)
    plt.legend()
    plt.show()
