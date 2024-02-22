# stgp_v01_2
# spatial-temporal GP model
# v01: use SE kernel for space (no elevation) and SE kernel for time
#      stData must NOT contians NaN values. Tips: if there are, use interpolation in fill them if the the missing values are minor.
# v01_2: add elevation as model input.
#        lat & long share the same length sacle, elevation use its own length scale
#
# v02: modual version of v01_2
# stgp_cali_v01: modify from stgp_v02 to add auto calibration

# %%
import torch
import torch.nn as nn
import numpy as np
import math
from matplotlib import pyplot as plt

# from KroneckerProduct import KroneckerProduct

def kronecker(A, B):
    AB = torch.einsum("ab,cd->acbd", A, B)
    AB = AB.view(A.size(0) * B.size(0), A.size(1) * B.size(1))
    return AB

# torch repeat
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


# combination uisng Kronecker product similar manner
def combinations(A, B):
    A1 = tile(A, 0, B.size(0))
    B1 = B.repeat(A.size(0), 1)
    # B1 = B.unsqueeze(0).expand(A.size(0),B.size(0), B.size(1)).reshape(A.size(0)*B.size(0), B.size(1)) Equivalent
    return torch.cat((A1, B1), dim=1)


JITTER = 1e-3
class stgp(nn.Module):
    def __init__(self, space_coordinates, time_coordinates, stData,
                 latlong_length_scale=4300., elevation_length_scale=30., time_length_scale=0.25,
                 noise_variance=0.1, signal_variance=1.):
        # space_coordinates musth a matrix of [number of space_coordinates x (lat,long,elevation)] in UTM or any meter coordinate.
        # time_coordinates musth a matrix of [number of time_coordinates x 1] in hour formate
        # stData musth be a matrix of [space_coordinates.size(0) x time_coordinates.size(0)]

        super(stgp, self).__init__()
        # input/output data
        self.space_coordinates = torch.tensor(space_coordinates)
        self.time_coordinates = torch.tensor(time_coordinates)
        self.stData = torch.tensor(stData)

        # print(space_length_scale)
        # a = space_length_scale * torch.ones(1)
        # print(a)
        # self.log_space_length_scale = nn.Parameter(torch.log( torch.tensor(space_length_scale) ))

        # kernel parameters
        self.log_latlong_length_scale = nn.Parameter(torch.log(torch.tensor(latlong_length_scale)), requires_grad = False)
        self.log_elevation_length_scale = nn.Parameter(torch.log(torch.tensor(elevation_length_scale)), requires_grad = False)
        self.log_time_length_scale = nn.Parameter(torch.log(torch.tensor(time_length_scale)), requires_grad = False)
        self.log_noise_variance = nn.Parameter(torch.log(torch.tensor(noise_variance)))
        self.log_signal_variance = nn.Parameter(torch.log(torch.tensor(signal_variance)))

        #calibration parameters
        self.bias = nn.Parameter(torch.zeros( self.space_coordinates.size(0),1))
        self.gain = nn.Parameter(torch.ones(1))
        self.stData_cali = self.stData

        self.update()

    def SE_kernel(self, X, X2, length_scale):
        # length_scale MUST be positive
        X = X / length_scale.expand(X.size(0), X.size(1))
        X2 = X2 / length_scale.expand(X2.size(0), X2.size(1))

        X_norm2 = torch.sum(X * X, dim=1).view(-1, 1)
        X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)

        # compute effective distance
        K = -2.0 * X @ X2.t() + X_norm2.expand(X.size(0), X2.size(0)) + X2_norm2.t().expand(X.size(0), X2.size(0))
        K = torch.exp(-K) * 1.0
        # K = self.log_scale.exp() * torch.exp(-K)
        return K

    def update(self):
        # update mean function, e.g., the calibration part
        # y_bias = self.bias.view(-1,1).repeat(self.time_coordinates.size(0) ,1)
        y_bias = self.bias.view(-1, 1).repeat(1,self.time_coordinates.size(0))
        self.stData_cali = self.stData * self.gain + y_bias

        # update kernels
        ndata = self.stData.numel()
        latlong_kernel = self.SE_kernel(self.space_coordinates[:, 0:2], self.space_coordinates[:, 0:2],
                                        torch.exp(self.log_latlong_length_scale))
        elevation_kernel = self.SE_kernel(self.space_coordinates[:, 2:3], self.space_coordinates[:, 2:3],
                                          torch.exp(self.log_elevation_length_scale))
        spatial_kernel = latlong_kernel * elevation_kernel + torch.eye(latlong_kernel.size(0)) * JITTER

        temporal_kernel = self.SE_kernel(self.time_coordinates, self.time_coordinates,
                                         torch.exp(self.log_time_length_scale)) + torch.eye(self.time_coordinates.size(0)) * JITTER

        eigen_value_s, eigen_vector_s = torch.symeig(spatial_kernel, eigenvectors=True)
        eigen_value_t, eigen_vector_t = torch.symeig(temporal_kernel, eigenvectors=True)

        eigen_vector_st = kronecker(eigen_vector_t, eigen_vector_s)
        eigen_value_st = kronecker(eigen_value_t.view(-1, 1), eigen_value_s.view(-1, 1)).view(-1)
        eigen_value_st_plus_noise_inverse = 1. / (eigen_value_st + torch.exp(self.log_noise_variance))

        # eigen_value_st_inverse = kronecker(torch.diag_embed(1/eigen_value_t), torch.diag_embed(1/eigen_value_s))
        # eigen_value_st_inverse += torch.eye(ndata) * 1/torch.exp(self.log_noise_variance)
        # eigen_value_st_inverse += torch.eye(ndata) * JITTER

        sigma_inverse = eigen_vector_st @ eigen_value_st_plus_noise_inverse.diag_embed() @ eigen_vector_st.transpose(-2,
                                                                                                                     -1)

        self.K = eigen_vector_st @ eigen_value_st.diag_embed() @ eigen_vector_st.transpose(-2, -1)
        # use .transpose(-2,-1) to be competible with batch data. Not necessary here.
        #
        # self.sigma1 = eigen_vector_st @ (eigen_value_st + torch.exp(self.log_noise_variance)).diag_embed()  @ eigen_vector_st.transpose(-2,-1)
        # self.sigma2 = self.K + torch.eye(ndata) * torch.exp(self.log_noise_variance)
        self.sigma_inverse = sigma_inverse
        # self.alpha = sigma_inverse @ self.stData.transpose(-2, -1).reshape(-1, 1)
        self.alpha = sigma_inverse @ self.stData_cali.transpose(-2, -1).reshape(-1, 1)
        self.eigen_value_st = eigen_value_st

    def forward(self, test_space_coordinates, test_time_coordinates):
        # with torch.no_grad():
        test_latlong_kernel = self.SE_kernel(test_space_coordinates[:, 0:2], self.space_coordinates[:, 0:2],
                                             torch.exp(self.log_latlong_length_scale))
        test_elevation_kernel = self.SE_kernel(test_space_coordinates[:, 2:3], self.space_coordinates[:, 2:3],
                                               torch.exp(self.log_elevation_length_scale))
        test_spatial_kernel = test_latlong_kernel * test_elevation_kernel

        test_temporal_kernel = self.SE_kernel(test_time_coordinates, self.time_coordinates,
                                              torch.exp(self.log_time_length_scale))

        test_st_kernel = kronecker(test_temporal_kernel, test_spatial_kernel)
        yPred = test_st_kernel @ self.alpha

        yVar = torch.zeros(test_st_kernel.size(0))
        for i in range(test_st_kernel.size(0)):
            yVar[i] = self.log_signal_variance.exp() - test_st_kernel[i:i + 1,
                                                       :] @ self.sigma_inverse @ test_st_kernel[i:i + 1, :].t()

        yPred = yPred.view(test_time_coordinates.size(0), test_space_coordinates.size(0)).transpose(-2, -1)
        # yPred = yPred.view(test_space_coordinates.size(0),test_time_coordinates.size(0))
        yVar = yVar.view(test_time_coordinates.size(0), test_space_coordinates.size(0)).transpose(-2, -1)
        return yPred, yVar

    def negative_log_likelihood(self):
        nll = 0
        nll += 0.5 * (self.eigen_value_st + torch.exp(self.log_noise_variance)).log().sum()
        # nll += 0.5 * (self.stData.transpose(-2, -1).reshape(1, -1) @ self.alpha).sum()
        nll += 0.5 * (self.stData_cali.transpose(-2, -1).reshape(1, -1) @ self.alpha).sum()
        # nll += 0.5 * self.stData.transpose(-2,-1).reshape(-1,1).t() @ self.sigma_inverse @ self.stData.transpose(
        # -2,-1).reshape(-1,1)
        return nll

    def train_bfgs(self, niteration, lr=0.001):
        # LBFGS optimizer
        optimizer = torch.optim.LBFGS(self.parameters(), lr=lr)  # lr is very important, lr>0.1 lead to failure
        for i in range(niteration):
            # optimizer.zero_grad()
            # LBFGS
            def closure():
                optimizer.zero_grad()
                self.update()
                loss = self.negative_log_likelihood()
                loss.backward()
                print('nll:', loss.item())
                return loss

            # optimizer.zero_grad()
            optimizer.step(closure)
            # print('loss:', loss.item())


    def train_adam(self, niteration=10, lr=0.001):
        # adam optimizer
        # uncommont the following to enable
        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        for i in range(niteration):
            optimizer.zero_grad()
            self.update()
            loss = self.negative_log_likelihood()
            loss.backward()
            optimizer.step()
            print('loss_nnl:', loss.item())

    # calibration training
    def cali_train_adam(self, space, time, Y_truth, niteration=10, lr=0.001):
        # adam optimizer
        # uncommont the following to enable
        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        for i in range(niteration):
            optimizer.zero_grad()
            self.update()
            loss_likelihood = self.negative_log_likelihood()

            ypred, yvar = self.forward(space, time)
            # prob = torch.distributions.multivariate_normal.MultivariateNormal(ypred.t().squeeze(), yvar.squeeze().diag_embed() )
            # loss_prediction = -prob.log_prob(Y_truth.t().squeeze())
            prob = torch.distributions.multivariate_normal.MultivariateNormal(ypred.flatten().float(),
                                                                              yvar.flatten().diag_embed().float())
            loss_prediction = -prob.log_prob(Y_truth.flatten().float())

            loss = loss_likelihood + loss_prediction
            #
            # if loss_prediction < 0:
            #     loss = loss_likelihood + loss_prediction
            # else:
            #     loss = loss_prediction
            loss.backward()
            optimizer.step()
            print('loss_total:', loss.item(), ' loss_likelihood:', loss_likelihood.item(), ' loss_prediction', loss_prediction.item() )

    def update_data(self, space_coordinates, time_coordinates, stData):
        self.space_coordinates = torch.tensor(space_coordinates)
        self.time_coordinates = torch.tensor(time_coordinates)
        self.stData = torch.tensor(stData)
        self.update()
