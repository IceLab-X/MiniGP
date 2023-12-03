import numpy as np
import torch
import torch.nn as nn
import kernel_v2 as kernel
import tensorly
import math
import os
import matplotlib.pyplot as plt
import unittest
from tensorly import tucker_to_tensor
tensorly.set_backend('pytorch')

class eigen_pairs:
    def __init__(self, matrix) -> None:
        eigen_value, eigen_vector = torch.linalg.eigh(matrix, UPLO='U')
        self.value = eigen_value
        self.vector = eigen_vector


class HOGP_simple(nn.Module):
    def __init__(self,noise_variance,output_shape,learnable_grid=False,learnable_map=False):
        super().__init__()
        self.noise_variance = nn.Parameter(torch.tensor([noise_variance]))
        self.kernel_list = []
        self.K = []
        self.K_eigen = []
        self.kernel_list = nn.ModuleList()
        for _ in range(len(output_shape) + 1):
            new_kernel = kernel.ARDKernel(1)
            self.kernel_list.append(new_kernel)
        self.grid = nn.ParameterList()  ##从单纯的列表创建到nn.ParameterList()
        for _value in output_shape:
            self.grid.append(nn.Parameter(torch.tensor(range(_value)).reshape(-1, 1).float()))
        if learnable_grid is False:
            for i in range(len(self.grid)):
                self.grid[i].requires_grad = False
        self.mapping_vector = nn.ParameterList()##从单纯的列表创建到nn.ParameterList()
        for _value in output_shape:
            self.mapping_vector.append(nn.Parameter(torch.eye(_value)))
        if learnable_map is False:
            for i in range(len(self.mapping_vector)):
                self.mapping_vector[i].requires_grad = False

        
    def forward(self,x_train,x_test):
        with torch.no_grad():
            
            K_star = self.kernel_list[0](x_test[0], x_train[0])
            K_predict = [K_star] + self.K[1:]

            predict_u = tensorly.tenalg.multi_mode_dot(self.g, K_predict)
            n_dim = len(self.K_eigen) - 1
            _init_value = torch.tensor([1.0]).reshape(*[1 for i in range(n_dim)]).to(x_train[0].device)
            diag_K_dims = tucker_to_tensor(( _init_value, [K.diag().reshape(-1,1) for K in self.K[1:]]))
            diag_K_dims = diag_K_dims.unsqueeze(0)
            diag_K_x = self.kernel_list[0](x_test[0], x_test[0]).diag()
            for i in range(n_dim):
                diag_K_x = diag_K_x.unsqueeze(-1)
            diag_K = diag_K_x*diag_K_dims

            S = self.A * self.A.pow(-1/2)
            S_2 = S.pow(2)
            eigen_vectors_x = K_star@self.K[0]
            eigen_vectors_dims = [self.K_eigen[i+1].vector.pow(2) for i in range(n_dim)]
            
            eigen_vectors = [eigen_vectors_x] + eigen_vectors_dims
            S_product = tensorly.tenalg.multi_mode_dot(S_2, eigen_vectors)
            var_diag = diag_K + S_product
        
        return predict_u, var_diag
    
    def log_likelihood(self, x_train,y_train):
        
        # for param_name, param in self.named_parameters():
        #     print(f"{param_name}: {param}")

        self.K.clear()
        self.K_eigen.clear()
        self.K.append(self.kernel_list[0](x_train[0], x_train[0]))
        self.K_eigen.append(eigen_pairs(self.K[-1]))

        for i in range(0, len(self.kernel_list)-1):
            _in = tensorly.tenalg.mode_dot(self.grid[i], self.mapping_vector[i], 0)
            self.K.append(self.kernel_list[i+1](_in, _in))
            self.K_eigen.append(eigen_pairs(self.K[-1]))
        _init_value = torch.tensor([1.0],  device=list(self.parameters())[0].device).reshape(*[1 for i in self.K])
        lambda_list = [eigen.value.reshape(-1, 1) for eigen in self.K_eigen]
        A = tucker_to_tensor((_init_value, lambda_list))
        A = A + self.noise_variance.pow(-1)* tensorly.ones(A.shape,  device=list(self.parameters())[0].device)

        T_1 = tensorly.tenalg.multi_mode_dot(y_train[0], [eigen.vector.T for eigen in self.K_eigen])
        T_2 = T_1 * A.pow(-1/2) 
        T_3 = tensorly.tenalg.multi_mode_dot(T_2, [eigen.vector for eigen in self.K_eigen]) 
        b = tensorly.tensor_to_vec(T_3)
        g = tensorly.tenalg.multi_mode_dot(T_1 * A.pow(-1), [eigen.vector for eigen in self.K_eigen]) 

        self.b = b
        self.A = A
        self.g= g
        nd = torch.prod(torch.tensor([value for value in self.A.shape]))
        loss = -1/2* nd * torch.log(torch.tensor(2 * math.pi, device=list(self.parameters())[0].device))
        loss += -1/2* torch.log(self.A).sum()
        loss += -1/2* self.b.t() @ self.b

        loss = -loss/nd
        return loss
    
    def to(self, *args, **kwargs): ##改写to方法，使kernel可以和模型一同移到同一个设备
        model = super().to(*args, **kwargs)
        for kernel in model.kernel_list:
            for param_name, param in kernel.named_parameters():
                param.to(*args, **kwargs)
        return model


class TestHOGP_simple(unittest.TestCase):

   def setUp(self):
       self.noise_variance = 0.01
       self.output_shape = (3, 4)
       self.model = HOGP_simple(self.noise_variance, self.output_shape)
       self.x_train = [torch.tensor(np.random.rand(10, 1), dtype=torch.float32)]
       self.y_train = [torch.tensor(np.random.rand(10, 1), dtype=torch.float32)]
       self.x_test = [torch.tensor(np.random.rand(5, 1), dtype=torch.float32)]

   def test_forward(self):
       # Test the forward pass
       # ...
       pass

   def test_log_likelihood(self):
       # Test the log likelihood calculation
       # ...
       pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    print("testing hogp")
    print(torch.__version__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device =torch.device('cpu')

    # single output test 1
    torch.manual_seed(1)       #set seed for reproducibility
    ##
    x = np.load('data_sample/input.npy')
    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = np.load('data_sample/output_fidelity_2.npy')
    y = torch.tensor(y, dtype=torch.float32).to(device)

    from utils.normalizer import Dateset_normalize_manager
    dnm = Dateset_normalize_manager([x], [y])

    xtr = [x[:128,:]]
    ytr = [y[:128,:]]
    xte = [x[128:,:]]
    yte = [y[128:,:]]

    output_shape=ytr[0][0,...].shape

    GPmodel=HOGP_simple(noise_variance=1.0,output_shape=output_shape)

    optimizer = torch.optim.Adam(GPmodel.parameters(), lr=1e-2)

    GPmodel.to(device)

    xtr, ytr = dnm.normalize_all(xtr, ytr)
    for i in range(300):
        optimizer.zero_grad()
        loss = GPmodel.log_likelihood(xtr, ytr)
        loss.backward()
        optimizer.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()))
        
    with torch.no_grad():
        xte=dnm.normalize_inputs(xte)
        ypred, ypred_var = GPmodel.forward(xtr,xte)
        ypred=dnm._denormalize(ypred,'output',0)
    
    ##plot_res_for_only_1
    # for i in range(yte[0].shape[0]):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    vmin=torch.min(yte[0][1])
    vmax=torch.max(yte[0][1])

    im=axs[0].imshow(yte[0][1].cpu(), cmap='hot', interpolation='nearest',vmin=vmin,vmax=vmax)
    axs[0].set_title('Groundtruth')

    axs[1].imshow(ypred[1].cpu(), cmap='hot', interpolation='nearest',vmin=vmin,vmax=vmax)
    axs[1].set_title('Predict')

    axs[2].imshow((yte[0][1].cpu()-ypred[1].cpu()).abs(), cmap='hot', interpolation='nearest',vmin=vmin,vmax=vmax)
    axs[2].set_title('Difference')

    cbar_ax = fig.add_axes([0.95, 0.2, 0.03, 0.6])
    cbar=fig.colorbar(im, cax=cbar_ax)
    output_filename = os.path.join("figures", f"figure.png")
    plt.savefig(output_filename)
    plt.close()
    # plt.show()
    
    ##Test_unit
    # unittest.main()
    
