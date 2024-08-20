import torch
import torch.nn as nn
import time
from core.kernel import NeuralKernel, ARDKernel
import core.GP_CommonCalculation as GP
JITTER= 1e-3
PI= 3.1415
torch.set_default_dtype(torch.float64)
class autoGP(nn.Module):
    def __init__(self, input_dim,device, kernel=None, inputwarp=False, num_inducing=None, deepkernel=False, training_size=None):
        super(autoGP, self).__init__()

        # GP hyperparameters
        self.log_beta = nn.Parameter(torch.ones(1) * 0) # Initial noise level

        self.inputwarp = inputwarp
        self.deepkernel = deepkernel
        self.device=device
        # Kernel setup
        if kernel is None:
            self.kernel = NeuralKernel(input_dim)
        else:
            self.kernel = kernel(input_dim)

        if input_dim > 2:
            self.deepkernel = True
            self.kernel = kernel(input_dim)

        if self.deepkernel:
            self.FeatureExtractor = torch.nn.Sequential(
                nn.Linear(input_dim, input_dim * 10),
                nn.LeakyReLU(),
                nn.Linear(input_dim * 10, input_dim * 5),
                nn.LeakyReLU(),
                nn.Linear(input_dim * 5, input_dim)
            ).to(device)
            self.inputwarp = False  # Disable inputwarp if deepkernel is enabled

        if self.inputwarp:
            self.warp = GP.Warp(method='kumar', initial_a=1.0, initial_b=1.0, warp_level=0.95).to(device)

        # Inducing points
        if num_inducing is None and training_size is not None:
            num_inducing = training_size * input_dim // 10
        if num_inducing is None and training_size is None:
            num_inducing = 100  # Default value if not specified
        self.xm = nn.Parameter(torch.rand((num_inducing, input_dim)))# Inducing points

    def negative_lower_bound(self, X, Y):
        """Negative lower bound as the loss function to minimize."""

        if self.deepkernel:
            X1 = self.FeatureExtractor(X)
            X = (X1 - X1.mean(0).expand_as(X1)) / X1.std(0).expand_as(X1)
        elif self.inputwarp:
            X = self.warp.transform(X)

        xm = self.xm

        n = X.size(0)
        K_mm = self.kernel(xm, xm) + JITTER * torch.eye(self.xm.size(0),device=self.device)
        L = torch.linalg.cholesky(K_mm)
        K_mn = self.kernel(xm, X)
        K_nn = self.kernel(X, X)
        A = torch.linalg.solve_triangular(L, K_mn, upper=False)
        A = A * torch.sqrt(self.log_beta.exp())
        AAT = A @ A.t()
        B =  AAT + (1+JITTER) * torch.eye(self.xm.size(0),device=self.device)
        LB = torch.linalg.cholesky(B)

        c = torch.linalg.solve_triangular(LB, A @ Y, upper=False)
        c = c * torch.sqrt(self.log_beta.exp())
        nll = (n / 2 * torch.log(2 * torch.tensor(PI)) +
               torch.sum(torch.log(torch.diagonal(LB))) +
               n / 2 * torch.log(1 / self.log_beta.exp()) +
               self.log_beta.exp() / 2 * torch.sum(Y * Y) -
               0.5 * torch.sum(c.squeeze() * c.squeeze()) +
               self.log_beta.exp() / 2 * torch.sum(torch.diagonal(K_nn)) -
               0.5 * torch.trace(AAT))
        return nll

    def optimal_inducing_point(self, X, Y):
        """Compute optimal inducing points mean and covariance."""
        if self.deepkernel:
            X1 = self.FeatureExtractor(X)
            X = (X1 - X1.mean(0).expand_as(X1)) / X1.std(0).expand_as(X1)
        elif self.inputwarp:
            X = self.warp.transform(X)

        xm = self.xm

        K_mm = self.kernel(xm, xm) + JITTER * torch.eye(self.xm.size(0),device=self.device)
        L = torch.linalg.cholesky(K_mm)
        L_inv = torch.inverse(L)
        K_mm_inv = L_inv.t() @ L_inv

        K_mn = self.kernel(xm, X)
        K_nm = K_mn.t()
        sigma = torch.inverse(K_mm + self.log_beta.exp() * K_mn @ K_nm)

        mean_m = self.log_beta.exp() * (K_mm @ sigma @ K_mn) @ Y
        A_m = K_mm @ sigma @ K_mm
        return mean_m, A_m, K_mm_inv

    def forward(self, X, Y, Xte):
        """Compute mean and variance for posterior distribution."""
        if self.deepkernel:
            X1 = self.FeatureExtractor(X)
            Xte1 = self.FeatureExtractor(Xte)
            Xte = (Xte1 - X1.mean(0).expand_as(Xte1)) / X1.std(0).expand_as(Xte1)
        elif self.inputwarp:
            Xte = self.warp.transform(Xte)

        xm = self.xm
        K_tt = self.kernel(Xte, Xte)
        K_tm = self.kernel(Xte, xm)
        K_mt = K_tm.t()
        mean_m, A_m, K_mm_inv = self.optimal_inducing_point(X, Y)
        mean = (K_tm @ K_mm_inv) @ mean_m
        var = (K_tt - K_tm @ K_mm_inv @ K_mt +
               K_tm @ K_mm_inv @ A_m @ K_mm_inv @ K_mt)
        var_diag = var.diag().view(-1, 1)

        return mean, var_diag

    def train_auto(self, X, Y, niteration1=180, lr1=0.1, niteration2=20, lr2=0.001):
        start_time = time.time()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr1)
        for i in range(niteration1):
            optimizer.zero_grad()
            loss = self.negative_lower_bound(X, Y)
            loss.backward()
            optimizer.step()
            #print(f'Iteration {i + 1}/{niteration1}, Loss: {loss.item()}')
        optimizer = torch.optim.LBFGS(self.parameters(), max_iter=niteration2, lr=lr2)

        def closure():
            optimizer.zero_grad()
            loss = self.negative_lower_bound(X, Y)
            loss.backward()

            return loss

        optimizer.step(closure)
        #print('loss:', loss.item())
        end_time = time.time()
        training_time = end_time - start_time
        print(f'AutoGP training completed in {training_time:.2f} seconds')


if __name__ == "__main__":
    from data_sample import generate_example_data as data
    xtr, ytr, xte, yte = data.generate(500, 500, seed=1, input_dim=1)
    device = torch.device('cuda')
    xtr, ytr, xte, yte = xtr.to(device), ytr.to(device), xte.to(device), yte.to(device)

    # Normalize the data outside the model
    normalizer = GP.DataNormalization(method='min_max').to(device)
    normalizer.fit(xtr, 'x')
    normalizer.fit(ytr, 'y')
    normalizer.fit(xte, 'xte')
    xtr_normalized = normalizer.normalize(xtr, 'x')
    ytr_normalized = normalizer.normalize(ytr, 'y')
    xte_normalized = normalizer.normalize(xte, 'xte')

    model = autoGP(input_dim=xtr_normalized.size(1), device=device, kernel=NeuralKernel, inputwarp=True,
                   deepkernel=False).to(device)

    model.train_auto(xtr_normalized, ytr_normalized)

    mean, var = model.forward(xtr_normalized, ytr_normalized, xte_normalized)
    mean=normalizer.denormalize(mean, 'y')
    var=normalizer.denormalize_cov(var, 'y')
    mse = torch.mean((mean - yte) ** 2)
    std=torch.sqrt(var)
    print(f'MSE for autoGP: {mse.item()}')
    print(model.warp.a,model.warp.b)
    import matplotlib.pyplot as plt
    plt.plot(xte.cpu().numpy(), yte.cpu().numpy(), label='True')
    plt.plot(xte.cpu().numpy(), mean.detach().cpu().numpy(), label='Predicted')

    plt.fill_between(xte.cpu().numpy().squeeze(), (mean - 1.96*std).cpu().detach().numpy().squeeze(),
                     (mean + 1.96*std).cpu().detach().numpy().squeeze(), alpha=0.3,label='95% Confidence Interval')
    plt.legend()
    plt.show()