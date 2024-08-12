import os
import torch
import torch.nn as nn
import torch.optim as optim
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Fixing strange error if run in MacOS
from matplotlib import pyplot as plt
from core.kernel import ARDKernel
import core.GP_CommonCalculation as GP
import data_sample.generate_example_data as data
from core.GP_CommonCalculation import conjugate_gradient as cg
from torch.utils.data import TensorDataset, DataLoader

# Constants
JITTER = 1e-3
PI = 3.1415
torch.manual_seed(4)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

class ParametricGP(nn.Module):
    def __init__(self, X, Y, num_inducing,batchsize=None):
        super(ParametricGP, self).__init__()
        self.normalizer= GP.DataNormalization(method='standard')
        self.normalizer.fit(X,'x')
        self.normalizer.fit(Y,'y')

        self.kernel = ARDKernel(1).to(device)
        self.num_inducing = num_inducing
        input_dim = X.size(1)
        self.X_all, self.Y_all = X, Y
        self.X_all = self.normalizer.normalize(self.X_all, 'x')
        self.Y_all = self.normalizer.normalize(self.Y_all, 'y')
        torch.manual_seed(seed=4)
        # Inducing points
        self.xm = nn.Parameter(torch.rand(num_inducing, input_dim, dtype=torch.float64).to(device))
        self.qu_mean = nn.Parameter(torch.rand(num_inducing, 1, dtype=torch.float64).to(device))
        self.jitter= JITTER
        self.device = device
        # Gaussian noise
        self.log_beta = nn.Parameter(torch.ones(1, dtype=torch.float64).to(device) * -4)

        self.batchsize = batchsize
        if self.batchsize is not None:
            # Create TensorDataset and DataLoader for minibatch training
            dataset = TensorDataset(self.X_all, self.Y_all)
            self.dataloader = DataLoader(dataset, batch_size=self.batchsize, shuffle=False)
            self.iterator = iter(self.dataloader)
        else:
            self.iterator = None


    def new_batch(self):
        if self.iterator is not None:
            try:
                X_batch, Y_batch = next(self.iterator)
            except StopIteration:
                # Reinitialize the iterator if it reaches the end
                self.iterator = iter(self.dataloader)
                X_batch, Y_batch = next(self.iterator)
            return X_batch, Y_batch
        else:
            return self.X_all, self.Y_all
    def predict(self, x):
        K_mm = self.kernel(self.xm, self.xm) + self.jitter * torch.eye(self.xm.size(0), dtype=torch.float64).to(
            self.device)
        K_tm = self.kernel(x, self.xm)
        y_pred = K_tm @ cg(K_mm, self.qu_mean)
        return y_pred

    def forward(self, xte):
        xte_normalized = self.normalizer.normalize(xte, 'x')
        K_mm = self.kernel(self.xm, self.xm) + self.jitter * torch.eye(self.xm.size(0), dtype=torch.float64).to(
            self.device)
        K_tm = self.kernel(xte_normalized, self.xm)
        K_tt_diag = self.kernel(xte_normalized, xte_normalized).diag()
        y_pred_normalized = K_tm @ cg(K_mm, self.qu_mean)
        y_var = K_tt_diag - (K_tm @ cg(K_mm, K_tm.t())).diag() + self.log_beta.exp().pow(-1)
        y_pred = self.normalizer.denormalize(y_pred_normalized, 'y')
        y_var = self.normalizer.denormalize_cov(y_var, 'y')
        y_var = y_var.view(-1, 1)
        return y_pred, y_var

    def loss_function(self,x_batch,y_batch):
        y_pred= self.predict(x_batch)
        n = y_batch.size(0)
        loss = 0.5 * n * (
            torch.log(self.log_beta.exp().pow(-1))+ torch.log(2 * torch.tensor(PI, dtype=torch.float64))) + \
               0.5 * (y_batch - y_pred).pow(2).sum() / self.log_beta.exp().pow(-1)
        return loss

if __name__ == '__main__':
    # Generate data
    xtr, ytr, xte, yte = data.generate(2000, 500, seed=2)
    xtr= xtr.to(device)
    ytr = ytr.to(device)
    xte = xte.to(device)
    yte= yte.to(device)

    # Training the model
    num_inducing = 100
    learning_rate = 0.1
    num_epochs = 800
    batchsize = 500
    # Create an instance of ParametricGP
    model = ParametricGP(xtr, ytr, num_inducing=num_inducing,batchsize=batchsize).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_values = []
    iteration_times = []
    import time
    for i in range(num_epochs):
        start_time = time.time()
        optimizer.zero_grad()
        X_batch, Y_batch = model.new_batch()
        loss = model.loss_function(X_batch, Y_batch)
        loss.backward()
        optimizer.step()
        end_time = time.time()

        loss_values.append(loss.item())
        iteration_times.append(end_time - start_time)

        if i % 10 == 0:  # Adjusted to show loss at every step for such a small number of epochs
            print('iter', i, 'nll:{:.5f}'.format(loss.item()))

    average_iteration_time = sum(iteration_times) / len(iteration_times)
    print(f'Average iteration time: {average_iteration_time:.5f} seconds')

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        ypred,yvar = model.forward(xte)
        mse = torch.mean((ypred - yte) ** 2)
        print(f'Test MSE: {mse.item()}')

    # Plotting the results
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(num_epochs), loss_values, label='Training Loss')
    # plt.xlabel('Iteration')
    # plt.ylabel('Negative Log-Likelihood Loss')
    # plt.title('Training Loss over Iterations')
    # plt.legend()
    # plt.show()

    plt.figure(figsize=(10, 6))
    #plt.plot(xtr.cpu().numpy(), ytr.cpu().numpy(), 'kx', label='Training data')
    normalizer=model.normalizer
    qu_mean=normalizer.denormalize(model.qu_mean.detach(),'y')
    xm=normalizer.denormalize(model.xm.cpu().detach(),'x')
    plt.scatter(xm.cpu(), qu_mean.cpu(), color='red', label='Inducing points')
    plt.plot(xte.cpu().numpy(), yte.cpu().numpy(), 'b', label='True function')
    plt.plot(xte.cpu().numpy(), ypred.cpu().numpy(), 'r', label='Predicted function')
    plt.fill_between(xte.cpu().numpy().squeeze(),
                     (ypred.cpu().numpy() - 1.96 * torch.sqrt(yvar).cpu().numpy()).squeeze(),
                     (ypred.cpu().numpy() + 1.96 * torch.sqrt(yvar).cpu().numpy()).squeeze(), alpha=0.2)
    plt.xlim(0,1)
    plt.legend()
    plt.show()
