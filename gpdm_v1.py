# ---- coding: utf-8 ----
# @author: Ziyang Zhang
# @version: v1, original GPDM with CUDA supported
#  - observationGP:    x->y, RBF
#  - dynamicGP:        x0->x1, RBF+Linear
# This work partly uses the code from CIGP and CGPDM.
# @license: (C) Copyright 2023, AMMLE Group Limited.


import torch
import numpy as np
import time
from sklearn.decomposition import PCA
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt

sigma_n_num_Y = 10 ** -3
sigma_n_num_X = 10 ** -3


class GPDM(torch.nn.Module):
    def __init__(self, D, Q, dyn_target):

        super(GPDM, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64

        # observation dimension
        self.D = D
        # latent dimension
        self.Q = Q
        # dynamic model target, choose full or delta
        self.dyn_target = dyn_target

        """ Set Y-kernel parameters (RBF) """
        # log_lengthscale in RBF kernel
        self.y_log_lengthscales = torch.nn.Parameter(
            torch.tensor(np.log(np.abs(np.ones(Q))), dtype=self.dtype, device=self.device))
        # log(signal inverse std), an initial scaling vector used for constructing W_y
        self.y_log_lambdas = torch.nn.Parameter(
            torch.tensor(np.log(np.abs(np.ones(D))), dtype=self.dtype, device=self.device))
        # log(noise std), noise in RBF kernel
        self.y_log_sigma_n = torch.nn.Parameter(
            torch.tensor(np.log(np.abs(np.ones(1))), dtype=self.dtype, device=self.device))

        """ Set X-kernel parameters (linear+RBF) """
        # log_lengthscale in RBF kernel
        self.x_log_lengthscales = torch.nn.Parameter(
            torch.tensor(np.log(np.abs(np.ones(Q))), dtype=self.dtype, device=self.device))
        # log(signal inverse std), an initial scaling vector used for constructing W_x
        self.x_log_lambdas = torch.nn.Parameter(
            torch.tensor(np.log(np.abs(np.ones(Q))), dtype=self.dtype, device=self.device))
        # # log(noise std), noise in RBF kernel
        self.x_log_sigma_n = torch.nn.Parameter(
            torch.tensor(np.log(np.abs(np.ones(1))), dtype=self.dtype, device=self.device))
        # log_linear_coefficients in linear kernel
        self.log_coe_linear = torch.nn.Parameter(torch.zeros(1, dtype=self.dtype, device=self.device))
        # additional noise variance for numerical issues
        self.sigma_n_num_Y = sigma_n_num_Y
        self.sigma_n_num_X = sigma_n_num_X

        # Initialize observations
        self.observations_list = []
        self.num_sequences = 0

    def add_data(self, Y):
        # Note: each sequence shares the same timestep N and dimension D
        if Y.shape[1] != self.D:
            raise ValueError('Y must be a N x D matrix collecting observation data!')
        self.observations_list.append(Y)
        self.num_sequences = self.num_sequences + 1
        print('Num. of sequences = ' + str(self.num_sequences) + ' [Data points = ' + str(
            np.concatenate(self.observations_list, 0).shape[0]) + ']')

    def observationGP_kernel(self, X1, X2, flg_noise=True):
        """
        currently use RBF
        """
        return self.kernel_rbf(X1, X2, self.y_log_lengthscales, self.y_log_sigma_n, self.sigma_n_num_Y, flg_noise)

    def dynamicGP_kernel(self, X1, X2, flg_noise=True):
        """
        currently use RBF+linear
        """
        return self.kernel_rbf(X1, X2, self.x_log_lengthscales, self.x_log_sigma_n, self.sigma_n_num_X, flg_noise) + \
               self.kernel_lin(X1, X2)

    def kernel_rbf(self, X1, X2, log_lengthscales, log_sigma_n, sigma_n_num=0, flg_noise=True):
        N = X1.shape[0]
        X1 = X1 / log_lengthscales.exp()
        X2 = X2 / log_lengthscales.exp()
        X1_norm2 = torch.sum(X1 * X1, dim=1).view(-1, 1)
        X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)
        if flg_noise:
            K = -2.0 * X1 @ X2.t() + X1_norm2.expand(X1.size(0), X2.size(0)) + X2_norm2.t().expand(X1.size(0),
                                                                                                   X2.size(0))
            return torch.exp(-0.5 * K) + torch.exp(log_sigma_n) ** 2 * torch.eye(N, dtype=self.dtype,
                                                                                 device=self.device) + \
                   sigma_n_num ** 2 * torch.eye(N, dtype=self.dtype, device=self.device)
        else:
            K = -2.0 * X1 @ X2.t() + X1_norm2.expand(X1.size(0), X2.size(0)) + X2_norm2.t().expand(X1.size(0),
                                                                                                   X2.size(0))
            return torch.exp(-0.5 * K)

    def kernel_lin(self, x1, x2):
        k_linear = self.log_coe_linear.exp() * (x1 @ x2.t())
        return k_linear

    def get_y_neg_log_likelihood(self, Y, X, N):
        """
        L_y = D/2*log(|K_y(X,X)|) + 1/2*trace(K_y^-1*Y*W_y^2*Y) - N*log(|W_y|)
        """
        K_y = self.observationGP_kernel(X, X)
        U, info = torch.linalg.cholesky_ex(K_y, upper=True)
        U_inv = torch.inverse(U)
        Ky_inv = U_inv @ U_inv.t()
        log_det_K_y = 2 * torch.sum(torch.log(torch.diag(U)))

        W2 = torch.diag(torch.exp(self.y_log_lambdas) ** 2)
        log_det_W = 2 * torch.sum(self.y_log_lambdas)

        return self.D / 2 * log_det_K_y + 1 / 2 * \
               torch.trace(torch.linalg.multi_dot([Ky_inv, Y, W2, Y.t()])) \
               - N * log_det_W

    def get_x_neg_log_likelihood(self, Xout, Xin, flg_W=True):
        """
        If use scaling matrix W, choose flg_W = True
        L_x = d/2*log(|K_x(Xin,Xin)|) + 1/2*trace(K_x^-1*Xout*W_x^2*Xout) - (N-dyn_back_step)*log(|W_x|)

        If use original GPDM, choose flg_W = False
        L_x = d/2*log(|K_x(Xin,Xin)|) + 1/2*trace(K_x^-1*Xout*Xout)

        ISSUE: W_x is necessary
        """

        K_x = self.dynamicGP_kernel(Xin, Xin)
        U, info = torch.linalg.cholesky_ex(K_x, upper=True)
        U_inv = torch.inverse(U)
        Kx_inv = U_inv @ U_inv.t()
        log_det_K_x = 2 * torch.sum(torch.log(torch.diag(U)))

        if flg_W:
            W2 = torch.diag(torch.exp(self.x_log_lambdas) ** 2)
            log_det_W = 2 * torch.sum(self.x_log_lambdas)
            return self.Q / 2 * log_det_K_x + 1 / 2 * \
                   torch.trace(torch.linalg.multi_dot([Kx_inv, Xout, W2, Xout.t()])) \
                   - Xin.shape[0] * log_det_W
        else:
            return self.Q / 2 * log_det_K_x + 1 / 2 * \
                   torch.trace(torch.linalg.multi_dot([Kx_inv, Xout, Xout.t()]))

    def get_Xin_Xout(self):
        X_list = []
        x_start_index = 0
        start_indeces = []
        for j in range(len(self.observations_list)):
            sequence_length = self.observations_list[j].shape[0]
            X_list.append(self.X[x_start_index:x_start_index + sequence_length, :])
            start_indeces.append(x_start_index)
            x_start_index = x_start_index + sequence_length

        if self.dyn_target == 'full':
            # in: x(t)
            Xin = X_list[0][:-1, :]
            # out: x(t+1)
            Xout = X_list[0][1:, :]
            for j in range(1, len(self.observations_list)):
                Xin = torch.cat((Xin, X_list[j][:-1, :]), 0)
                Xout = torch.cat((Xout, X_list[j][1:, :]), 0)
        elif self.dyn_target == 'delta':
            # in: x(t)
            Xin = X_list[0][:-1, :]
            # out: x(t+1)-x(t)
            Xout = X_list[0][1:, :] - X_list[0][:-1, :]
            for j in range(1, len(self.observations_list)):
                Xin = torch.cat((Xin, X_list[j][:-1, :]), 0)
                Xout = torch.cat((Xout, X_list[j][1:, :] - X_list[j][0:-1, :]), 0)
        else:
            raise ValueError('target must be either \'full\' or \'delta\'')

        return Xin, Xout

    def gpdm_loss(self, Y, N, balance=1):
        """
        loss = Ly + B*Lx
        """
        Xin, Xout = self.get_Xin_Xout()
        lossY = self.get_y_neg_log_likelihood(Y, self.X, N)
        lossX = self.get_x_neg_log_likelihood(Xout, Xin)
        loss = lossY + balance * lossX
        return loss

    def init_X(self):
        """
        initialize latent embeddings X with PCA
        return the latent trajectories associated to each observation sequence recorded
        """
        Y = np.concatenate(self.observations_list, 0)  # (M*N, D)
        pca = PCA(n_components=self.Q)
        X0 = pca.fit_transform(Y)

        # set latent variables as parameters
        self.X = torch.nn.Parameter(torch.tensor(X0, dtype=self.dtype, device=self.device), requires_grad=True)

        # init inverse kernel matrices
        Ky = self.observationGP_kernel(self.X, self.X)
        U, info = torch.linalg.cholesky_ex(Ky, upper=True)
        U_inv = torch.inverse(U)
        self.Ky_inv = U_inv @ U_inv.t()

        Xin, Xout = self.get_Xin_Xout()
        Kx = self.dynamicGP_kernel(Xin, Xin)
        U, info = torch.linalg.cholesky_ex(Kx, upper=True)
        U_inv = torch.inverse(U)
        self.Kx_inv = U_inv @ U_inv.t()

        return self.get_latent_sequences()

    def get_latent_sequences(self):
        X_np = self.X.clone().detach().cpu().numpy()
        X_np = X_np.reshape(len(self.observations_list), -1, self.X.shape[1])
        return [x for x in X_np]

    def train_lbfgs(self, num_opt_steps, lr=0.01, balance=1):
        print('\n### START TRAINING WITH L-BFGS ###')
        Y = torch.tensor(np.concatenate(self.observations_list, 0), dtype=self.dtype, device=self.device)  # (M*N, D)
        N = Y.shape[0]
        # optimizer
        optimizer = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=20, history_size=7,line_search_fn='strong_wolfe')
        # training
        losses = []
        t_start = time.time()
        for epoch in range(num_opt_steps):
            def closure():
                optimizer.zero_grad()
                loss = self.gpdm_loss(Y, N, balance)
                if loss.requires_grad:
                    loss.backward()
                return loss

            losses.append(closure().item())
            optimizer.step(closure)

            print('\nEpoch:' + str(epoch+1) + '/' + str(num_opt_steps))
            print('Running loss:', "{:.4e}".format(losses[-1]))
            t_stop = time.time()
            print('Used time:', t_stop - t_start)
            t_start = t_stop

        # save inverse kernel matrices of dynamicGP after training
        # calculate in forward_dynamicGP() will be extremely slow
        Xin, Xout = self.get_Xin_Xout()
        U, _ = torch.linalg.cholesky_ex(self.dynamicGP_kernel(Xin, Xin), upper=True)
        U_inv = torch.inverse(U)
        self.Kx_inv = U_inv @ U_inv.t()

        return losses

    def forward_observationGP(self, Xstar, flg_noise=False):
        U, _ = torch.linalg.cholesky_ex(self.observationGP_kernel(self.X, self.X), upper=True)
        U_inv = torch.inverse(U)
        Ky_inv = U_inv @ U_inv.t()
        Y_obs = torch.tensor(np.concatenate(self.observations_list, 0), dtype=self.dtype,
                             device=self.device)  # (M*N, D)
        Ky_star = self.observationGP_kernel(self.X, Xstar, False)
        mean_Y_pred = torch.linalg.multi_dot([Y_obs.t(), Ky_inv, Ky_star]).t()
        if flg_noise:
            diag_var_Y_pred_common = torch.ones(Xstar.shape[0], dtype=self.dtype, device=self.device) + \
                                     torch.exp(self.y_log_sigma_n) ** 2 + self.sigma_n_num_Y ** 2
        else:
            diag_var_Y_pred_common = torch.ones(Xstar.shape[0], dtype=self.dtype, device=self.device)
        diag_var_Y_pred_common = diag_var_Y_pred_common - torch.sum(Ky_star.t() @ Ky_inv * Ky_star.t(),
                                                                    dim=1)
        y_log_lambdas = torch.exp(self.y_log_lambdas) ** -2
        diag_var_Y_pred = diag_var_Y_pred_common.unsqueeze(1) * y_log_lambdas.unsqueeze(0)

        return mean_Y_pred, diag_var_Y_pred

    def forward_dynamicGP(self, Xstar, flg_noise=False):
        Xin, Xout = self.get_Xin_Xout()
        n = Xstar.shape[0]
        Kx_star = self.dynamicGP_kernel(Xin, Xstar, False)
        mean_Xout_pred = torch.linalg.multi_dot([Xout.t(), self.Kx_inv, Kx_star]).t()
        if flg_noise:
            diag_var_Xout_pred_common = torch.ones(n, dtype=self.dtype, device=self.device) + \
                                        torch.exp(self.x_log_sigma_n) ** 2 + self.sigma_n_num_X ** 2 + \
                                        torch.sum(self.log_coe_linear.exp() * Xstar * Xstar) - \
                                        torch.sum(Kx_star.t() @ self.Kx_inv * Kx_star.t(), dim=1)
        else:
            diag_var_Xout_pred_common = torch.ones(n, dtype=self.dtype, device=self.device) + \
                                        torch.sum(self.log_coe_linear.exp() * Xstar * Xstar) - \
                                        torch.sum(Kx_star.t() @ self.Kx_inv * Kx_star.t(), dim=1)
        x_log_lambdas = torch.exp(self.x_log_lambdas) ** -2
        diag_var_Xout_pred = diag_var_Xout_pred_common.unsqueeze(1) * x_log_lambdas.unsqueeze(0)

        return mean_Xout_pred, diag_var_Xout_pred

    def forward(self, num_steps, X0, num_sample=10, flg_noise=False):
        print('\n ### START SAMPLING & PREDICTING... ###')
        with torch.no_grad():
            X_hat = torch.zeros((num_steps, self.Q), dtype=self.dtype, device=self.device)
            # init latent variables
            X_hat[0, :] = torch.tensor(X0, dtype=self.dtype, device=self.device)
            t_start = 0
            # sample dynamic GP
            sample_list = torch.zeros((num_sample, num_steps, self.Q), dtype=self.dtype, device=self.device)
            # generate latent rollout
            for t in range(t_start, num_steps):
                Xin = X_hat[t:t + 1, :]
                mean_Xout_pred, var_Xout_pred = self.forward_dynamicGP(Xin, flg_noise)
                # generate distribution to sample
                distribution = Normal(mean_Xout_pred, torch.sqrt(var_Xout_pred))
                sample_list[:, t_start, :] = distribution.sample((num_sample,)).squeeze(1)

                if self.dyn_target == 'full':
                    X_hat[t + 1:t + 2, :] = mean_Xout_pred
                elif self.dyn_target == 'delta':
                    X_hat[t + 1:t + 2, :] = X_hat[t:, :] + mean_Xout_pred

            # map X mean to observation space to get Y mean
            mean_Y_pred, _ = self.forward_observationGP(X_hat, flg_noise)
            # map X samples to observation space to get Y var
            var_list = torch.zeros((num_sample, num_steps, self.D), dtype=self.dtype, device=self.device)
            for s in range(num_sample):
                X_mean_sample = sample_list[s, :, :]
                _, var_Y_sample = self.forward_observationGP(X_mean_sample, flg_noise)
                var_list[s, :, :] = var_Y_sample
            # get samples mean
            var_Y_pred = torch.mean(var_list, dim=0)

            return X_hat.detach().cpu().numpy(), mean_Y_pred.detach().cpu().numpy(), var_Y_pred.detach().cpu().numpy()


if __name__ == "__main__":

    """ hyper-parameters """
    Q = 3  # latent dim
    epochs = 3  # optimization steps (max epochs)
    lr = 0.01  # learning rate

    print('\n *********** TRAIN GPDM *********** :')
    print(' - latent dimension: ' + str(Q))
    print(' - optimization steps: ' + str(epochs))
    print(' - learning rate: ' + str(lr))
    print(' - device: ' + "cuda" if torch.cuda.is_available() else "cpu")
    print('\n')


    """ prepare data """
    # generate periodic data (observation data)
    # here we generate 5 sequences, each with (N=200, D=5)
    Y_data = []
    for i in range(5):
        i += 1
        y1 = np.sin(np.arange(0, 20, 0.1)) * i
        y2 = np.cos(np.arange(0, 20, 0.1)) * 5 / i
        y3 = np.sin(np.arange(0, 20, 0.1) + np.pi / 4 * i / 2) * 4
        y4 = np.cos(np.arange(0, 20, 0.1) + np.pi / 4 * i / 3) * 3
        y5 = np.sin(np.arange(0, 20, 0.1) + np.pi / 2 * i / 2) * 2 * i
        Y_data.append(np.concatenate(
            (y1.reshape(-1, 1), y2.reshape(-1, 1), y3.reshape(-1, 1), y4.reshape(-1, 1), y5.reshape(-1, 1)), axis=1))

    # plot training data
    fig, axs = plt.subplots(5, 1, figsize=(8, 8))
    fig.suptitle('Training Data, 5 seqs with shape (200x5)', fontsize=16)
    for i in range(5):
        for j in range(5):
            axs[i].plot([i for i in range(200)], Y_data[i][:,j])
    plt.show()


    """ init GPDM """
    D = Y_data[0].shape[1]
    dyn_target = 'full'  # choose full or delta, see Higher-order Features in the GPDM paper
    model = GPDM(D=D, Q=Q, dyn_target=dyn_target)

    # add training data
    for i in Y_data:
        model.add_data(i)

    # get initial X by PCA
    X_list_pca = model.init_X()


    """ train GPDM """
    start_time = time.time()
    loss = model.train_lbfgs(num_opt_steps=epochs, lr=lr, balance=1)
    end_time = time.time()
    train_time = end_time - start_time
    print("\nTotal Training Time: " + str(train_time) + " s")


    """ plot results """
    ## plot loss
    plt.figure()
    plt.plot(loss)
    plt.grid()
    plt.title('Loss')
    plt.xlabel('Optimization steps')
    plt.show()


    ## latent trajectories
    X_list = model.get_latent_sequences()
    plt.figure()
    plt.suptitle('Latent trajectories')
    for j in range(Q):
        plt.subplot(Q,1,j+1)
        plt.xlabel('Time [s]')
        plt.ylabel(r'$x_{'+str(j+1)+'}$')
        for i in range(len(X_list)):
            plt.plot(X_list[i][:,j])
        plt.grid()
    plt.show()

    ## latent trajectories in 3D (only when Q=3)
    if Q == 3:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = X_list[0]
        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]
        ax.plot(x, y, z, label='trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D trajectory')
        plt.show()


    ## test and plot one of the dimensions
    X_list = model.get_latent_sequences()
    Y_list = model.observations_list
    # choose the end of the sequences
    X = X_list[4]
    Y = Y_list[4]
    N = Y.shape[0]  # timestep
    forward_steps = 100  # how many steps to inference

    _, Ypred, Ystd = model(num_steps=forward_steps, num_sample=100, X0=X[-1, :], flg_noise=True)

    plt.figure()
    plt.plot([i for i in range(N)], Y[:, 0])  # original seq
    plt.plot([i + N for i in range(forward_steps)], Ypred[:, 0])  # inference part
    plt.fill_between([i + N for i in range(forward_steps)],       # confidence
                     Ypred[:, 0] + np.sqrt(Ystd[:, 0]),
                     Ypred[:, 0] - np.sqrt(Ystd[:, 0]), alpha=0.2)
    plt.show()

