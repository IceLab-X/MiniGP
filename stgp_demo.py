import torch
import torch.nn as nn
from scipy.io import loadmat
import matplotlib.pyplot as plt
JITTER = 1e-1

def kronecker(A, B):
    AB = torch.einsum("ab,cd->acbd", A, B)
    AB = AB.view(A.size(0) * B.size(0), A.size(1) * B.size(1))
    return AB

class SE_kernel(nn.Module):
    def __init__(self, length_scale=1.0):
        super().__init__()
        self.length_scale = torch.exp(nn.Parameter(torch.log(torch.tensor(length_scale))))
    def forward(self, X, X2):
        X = X / self.length_scale.expand(X.size(0), X.size(1))
        X2 = X2 / self.length_scale.expand(X2.size(0), X2.size(1))

        X_norm2 = torch.sum(X * X, dim=1).view(-1, 1)
        X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)

        # compute effective distance
        K = -2.0 * X @ X2.t() + X_norm2.expand(X.size(0), X2.size(0)) + X2_norm2.t().expand(X.size(0), X2.size(0))
        K = torch.exp(-K) * 1.0
        return K

class stgp(nn.Module):
    def __init__(self, latlong_length_scale=4300., elevation_length_scale=30., time_length_scale=0.25,
                 noise_variance=0.1, signal_variance=1.):
        super(stgp, self).__init__()

        self.log_noise_variance = nn.Parameter(torch.log(torch.tensor(noise_variance)))
        self.log_signal_variance = nn.Parameter(torch.log(torch.tensor(signal_variance)))

        self.latlong_kernel = SE_kernel(latlong_length_scale)
        self.elevation_kernel = SE_kernel(elevation_length_scale)
        self.temporal_kernel = SE_kernel(time_length_scale)
        
    def negative_log_likelihood(self, space_coordinates, time_coordinates, stData):

        latlong_K = self.latlong_kernel(space_coordinates[:, 0:2], space_coordinates[:, 0:2])
        elevation_K = self.elevation_kernel(space_coordinates[:, 2:3], space_coordinates[:, 2:3])

        spatial_K = latlong_K * elevation_K + torch.eye(latlong_K.size(0)) * JITTER
        temporal_K = self.temporal_kernel(time_coordinates, time_coordinates) + torch.eye(time_coordinates.size(0)) * JITTER

        eigen_value_s, eigen_vector_s = torch.linalg.eigh(spatial_K, UPLO='L')
        eigen_value_t, eigen_vector_t = torch.linalg.eigh(temporal_K, UPLO='L')

        eigen_vector_st = kronecker(eigen_vector_t, eigen_vector_s)
        eigen_value_st = kronecker(eigen_value_t.view(-1, 1), eigen_value_s.view(-1, 1)).view(-1)
        eigen_value_st_plus_noise_inverse = 1. / (eigen_value_st + torch.exp(self.log_noise_variance))

        sigma_inverse = eigen_vector_st @ eigen_value_st_plus_noise_inverse.diag_embed() @ eigen_vector_st.transpose(-2, -1)

        # self.K = eigen_vector_st @ eigen_value_st.diag_embed() @ eigen_vector_st.transpose(-2, -1)
        
        # detach.cpu()?
        self.sigma_inverse = sigma_inverse.detach().cpu()
        self.alpha = sigma_inverse @ stData.transpose(-2, -1).reshape(-1, 1).detach().cpu()

        nll = 0
        nll += 0.5 * (eigen_value_st + torch.exp(self.log_noise_variance)).log().sum()
        nll += 0.5 * (stData.transpose(-2, -1).reshape(1, -1) @ self.alpha).sum()
        return nll

    def forward(self, train_space_coordinates, train_time_coordinates, test_space_coordinates, test_time_coordinates):

        test_latlong_K = self.latlong_kernel(test_space_coordinates[:, 0:2], train_space_coordinates[:, 0:2])
        test_elevation_K = self.elevation_kernel(test_space_coordinates[:, 2:3], train_space_coordinates[:, 2:3])
        test_spatial_K = test_latlong_K * test_elevation_K
        test_temporal_K = self.temporal_kernel(test_time_coordinates, train_time_coordinates)
        test_st_K = kronecker(test_temporal_K, test_spatial_K)
        yPred = test_st_K @ self.alpha
        yVar = torch.zeros(test_st_K.size(0))
        for i in range(test_st_K.size(0)):
            yVar[i] = self.log_signal_variance.exp() - test_st_K[i:i + 1,:] @ self.sigma_inverse @ test_st_K[i:i + 1, :].t()
        yPred = yPred.view(test_time_coordinates.size(0), test_space_coordinates.size(0)).transpose(-2, -1)
        yVar = yVar.view(test_time_coordinates.size(0), test_space_coordinates.size(0)).transpose(-2, -1)
        
        return yPred, yVar

def train_stgp(model, train_space_coordinates, train_time_coordinates, train_stData, lr=0.1, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.LBFGS(modelparameters(), lr=lr)
    for i in range(epochs):
        optimizer.zero_grad()
        loss = model.negative_log_likelihood(train_space_coordinates, train_time_coordinates, train_stData)
        loss.backward()
        optimizer.step()
        print('Epoch: {}, Loss: {}'.format(i, loss.item()))


if __name__ == '__main__':
    data = loadmat('1_10_TODO\\data_v1_09_05_fill.mat') 
    time = data['time']
    str = data['sTr'] 
    pm25 = data['pm25']
    str_daq = data['sTr_DAQ']
    pm25_daq = data['pm25_daq']

    str = torch.tensor(str)
    time = torch.tensor(time)  
    pm25 = torch.tensor(pm25)

    model = stgp()
    # nll1 = model.negative_log_likelihood(str, time, pm25)
    str_daq = torch.tensor(str_daq)
    # with torch.no_grad():
    #     yPred, yVar = model(str, time, str_daq, time)

    # plt.plot(yPred.T,'--')  #show the predictions at the daq locations
    # plt.plot(pm25_daq.T,'-')    #show the daq observations
    # plt.show()

    train_stgp(model, str, time, pm25, lr=0.1, epochs=5)
    with torch.no_grad():
        yPred, yVar = model(str, time, str_daq, time)

    plt.plot(yPred.T,'--')  #show the predictions at the daq locations
    plt.plot(pm25_daq.T,'-')    #show the daq observations
    plt.show()
    plt.savefig('1_10_TODO\\stgp.png')
    pass
