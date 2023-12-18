# auto regression for multi-fidelity fusion.
# This function assumes that the high fidelity input is a superset of the low fidelity input.
# author: Wei Xing
# date: 2023-12-12
# version: 1.0

import numpy as np
import torch
import torch.nn as nn
import kernel as kernel
import MF_pack as mf
# import MINIGP.MultiTaskGP_cigp as CIGP
from gp_basic import GP_basic as CIGP
# from MultiTaskGP_cigp import cigp

# TODO: this codes needs to be improved for speed and memory usage
def encode_rows(matrix):
    """Encode rows of a matrix as strings for set operations."""
    return [','.join(map(str, row.tolist())) for row in matrix]

def find_matrix_row_overlap_and_indices(x_low, x_high):
    # Encode rows of both matrices
    encoded_x_low = encode_rows(x_low)
    encoded_x_high = encode_rows(x_high)

    # Find overlapping encoded rows
    overlap_set = set(encoded_x_low).intersection(encoded_x_high)

    # Get indices of overlapping rows
    overlap_indices_low = [i for i, row in enumerate(encoded_x_low) if row in overlap_set]
    overlap_indices_high = [i for i, row in enumerate(encoded_x_high) if row in overlap_set]

    return overlap_set, overlap_indices_low, overlap_indices_high


def GP_train(GPmodel, xtr, ytr, lr=1e-1, max_iter=1000, verbose=True):
    optimizer = torch.optim.Adam(GPmodel.parameters(), lr=1e-1)
    for i in range(max_iter):
        optimizer.zero_grad()
        loss = -GPmodel.log_likelihood(xtr, ytr)
        loss.backward()
        optimizer.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()))

    
class autoRegression(nn.Module):
    # initialize the model
    def __init__(self, gp):
        super(autoRegression, self).__init__()
        
        # create the model
        kernel1 = kernel.SumKernel(kernel.LinearKernel(1), kernel.MaternKernel(1))
        self.low_fidelity_GP = CIGP(kernel=kernel1, noise_variance=1.0)
        kernel2 = kernel.SumKernel(kernel.LinearKernel(1), kernel.MaternKernel(1))
        self.high_fidelity_GP = CIGP(kernel=kernel2, noise_variance=1.0)
        self.rho = nn.Parameter(torch.tensor(1))
    
    # define the forward pass
    def train(self, x_train, y_train, x_test):
        # get the data
        x_low = x_train[0]
        y_low = y_train[0]
        x_high = x_train[1]
        y_high = y_train[1]
        
        # train the low fidelity GP
        GP_train(self.low_fidelity_GP, x_low, y_low, lr=1e-1, max_iter=1000, verbose=True)
        
        # get the high fidelity part that is subset of the low fidelity part
        overlap_set, overlap_indices_low, overlap_indices_high = find_matrix_row_overlap_and_indices(x_low, x_high)
        
        # train the high fidelity GP
        optimizer = torch.optim.Adam(self.high_fidelity_GP.parameters(), lr=1e-1)
        for i in range(1000):
            optimizer.zero_grad()
            # y_residual = y_high[overlap_indices_high,:] - self.rho * self.low_fidelity_GP.predict(x_high[overlap_indices_high,:])
            y_residual = y_high[overlap_indices_high,:] - self.rho * y_low[overlap_indices_low,:]
            loss = -self.high_fidelity_GP.log_likelihood(overlap_set,y_residual)
            loss.backward()
            optimizer.step()
            print('iter', i, 'nll:{:.5f}'.format(loss.item()))
        
    def forward(self, x_test):
        # predict the model
        y_pred_low, cov_pred_low = self.low_fidelity_GP(x_test)
        y_pred_res, cov_pred_res= self.high_fidelity_GP(x_test)
        
        y_pred_high = y_pred_low + self.rho * y_pred_res
        cov_pred_high = cov_pred_low + (self.rho **2) * cov_pred_res
        
        # return the prediction
        return y_pred_low, cov_pred_high
        