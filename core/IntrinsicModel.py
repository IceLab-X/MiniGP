import torch
import torch.nn as nn
from scipy.io import loadmat
from core.kernel import SquaredExponentialKernel
import matplotlib.pyplot as plt
JITTER = 1e-1

def kronecker(A, B):
    """
    Computes the Kronecker product of two matrices A and B.

    Args:
        A (torch.Tensor): The first matrix of shape (m, n).
        B (torch.Tensor): The second matrix of shape (p, q).

    Returns:
        torch.Tensor: The Kronecker product of A and B, with shape (m * p, n * q).
    """
    AB = torch.einsum("ab,cd->acbd", A, B)
    AB = AB.view(A.size(0) * B.size(0), A.size(1) * B.size(1))
    return AB

class stgp(nn.Module):
    def __init__(self, latlong_length_scale=4300., elevation_length_scale=30., time_length_scale=0.25,
                     noise_variance=0.1, signal_variance=1.):
        """
        Initialize the MultiTaskGP_IMC class.

        Args:
            latlong_length_scale: Length scale for the latitude and longitude kernel.
            elevation_length_scale: Length scale for the elevation kernel.
            time_length_scale: Length scale for the temporal kernel.
            noise_variance: Variance of the noise.
            signal_variance: Variance of the signal.
        """
        super(stgp, self).__init__()

        self.log_noise_variance = nn.Parameter(torch.log(torch.tensor(noise_variance)))
        self.log_signal_variance = nn.Parameter(torch.log(torch.tensor(signal_variance)))

        self.latlong_kernel = SquaredExponentialKernel(length_scale = latlong_length_scale)
        self.elevation_kernel = SquaredExponentialKernel(length_scale = elevation_length_scale)
        self.temporal_kernel = SquaredExponentialKernel(length_scale = time_length_scale)

    def negative_log_likelihood(self, space_coordinates, time_coordinates, stData):
        """
        Calculates the negative log-likelihood of the MultiTaskGP_IMC model.

        Args:
            space_coordinates (torch.Tensor): Tensor containing the space coordinates.
            time_coordinates (torch.Tensor): Tensor containing the time coordinates.
            stData (torch.Tensor): Tensor containing the data.

        Returns:
            The negative log-likelihood value.
        """
        latlong_K = self.latlong_kernel(space_coordinates[:, 0:2], space_coordinates[:, 0:2])
        elevation_K = self.elevation_kernel(space_coordinates[:, 2:3], space_coordinates[:, 2:3])

        spatial_K = latlong_K * elevation_K + torch.eye(latlong_K.size(0)) * JITTER
        temporal_K = self.temporal_kernel(time_coordinates, time_coordinates) + torch.eye(time_coordinates.size(0)) * JITTER

        # Matrix Eigendecomposition
        eigen_value_s, eigen_vector_s = torch.linalg.eigh(spatial_K, UPLO='L')
        eigen_value_t, eigen_vector_t = torch.linalg.eigh(temporal_K, UPLO='L')

        eigen_vector_st = kronecker(eigen_vector_t, eigen_vector_s)
        eigen_value_st = kronecker(eigen_value_t.view(-1, 1), eigen_value_s.view(-1, 1)).view(-1)
        eigen_value_st_plus_noise_inverse = 1. / (eigen_value_st + torch.exp(self.log_noise_variance))

        Lambda_st = eigen_value_st_plus_noise_inverse.diag_embed()
        A = torch.flatten(eigen_vector_t.transpose(-2, -1) @ stData.transpose(-2, -1) @ eigen_vector_s).unsqueeze(-1)        
        self.alpha = Lambda_st @ A
        self.Lambda_st = Lambda_st
        self.eigen_vector_st = eigen_vector_st

        nll = 0
        nll += 0.5 * (eigen_value_st + torch.exp(self.log_noise_variance)).log().sum()
        nll += 0.5 * (A.transpose(-2, -1) @ self.alpha).sum()
        return nll

    def forward(self, train_space_coordinates, train_time_coordinates, test_space_coordinates, test_time_coordinates):
        """
        Performs the forward pass of the MultiTaskGP_IMC model.

        Args:
            train_space_coordinates (torch.Tensor): The space coordinates of the training data.
            train_time_coordinates (torch.Tensor): The time coordinates of the training data.
            test_space_coordinates (torch.Tensor): The space coordinates of the test data.
            test_time_coordinates (torch.Tensor): The time coordinates of the test data.

        Returns:
            yPred (torch.Tensor): The predicted output values.
            yVar (torch.Tensor): The variance of the predicted output values.
        """
        test_latlong_K = self.latlong_kernel(test_space_coordinates[:, 0:2], train_space_coordinates[:, 0:2])
        test_elevation_K = self.elevation_kernel(test_space_coordinates[:, 2:3], train_space_coordinates[:, 2:3])
        test_spatial_K = test_latlong_K * test_elevation_K
        test_temporal_K = self.temporal_kernel(test_time_coordinates, train_time_coordinates)
        test_st_K = kronecker(test_temporal_K, test_spatial_K)
        yPred = test_st_K @ self.eigen_vector_st @self.alpha

        sigma_inverse = self.eigen_vector_st @ self.Lambda_st @ self.eigen_vector_st.transpose(-2, -1)

        K_space_star2 = self.latlong_kernel(test_space_coordinates[:, 0:2], test_space_coordinates[:, 0:2])
        K_time_star2 = self.temporal_kernel(test_time_coordinates, test_time_coordinates)
        K_space_time_star2 = kronecker(K_time_star2, K_space_star2)
        yVar = K_space_time_star2.diag() - (test_st_K @ sigma_inverse @ test_st_K.t()).diag()

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
    data = loadmat('D:\iceLab\data_sample\data_v1_09_05_fill.mat') # you have to put your abosulte path here
    #time for the pm25 measurement data pm25 
    time = data['time']
    #spatial coordinate [lat,long,elevation] for all sensors in SLC
    str = data['sTr']
    #pm25 observed values for all sensors 
    pm25 = data['pm25']
    #spatial coordinate for daq sensors
    str_daq = data['sTr_DAQ']
    #reading from daq sensors
    pm25_daq = data['pm25_daq']

    str = torch.tensor(str)
    time = torch.tensor(time)  
    pm25 = torch.tensor(pm25)
    str_daq = torch.tensor(str_daq)

    model = stgp()

    train_stgp(model, str, time, pm25, lr=0.1, epochs=100)
    with torch.no_grad():
        yPred, yVar = model(str, time, str_daq, time)

    plt.plot(yPred.cpu().numpy().T, '--')  # show the predictions at the daq locations
    plt.plot(pm25_daq.T, '-')  # show the daq observations
    plt.show()
    # plt.savefig('1_10_TODO\\stgp.png')