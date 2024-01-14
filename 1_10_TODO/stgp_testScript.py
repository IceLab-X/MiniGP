# stgp_script

# %% data
import torch 
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

import stgp_v01

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

def kronecker(A, B):
    AB = torch.einsum("ab,cd->acbd", A, B)
    AB = AB.view(A.size(0)*B.size(0), A.size(1)*B.size(1))
    return AB

def combinations(A,B): 
    A1 = tile(A,0,B.size(0))
    B1 = B.repeat(A.size(0), 1)
    # B1 = B.unsqueeze(0).expand(A.size(0),B.size(0), B.size(1)).reshape(A.size(0)*B.size(0), B.size(1)) Equivalent
    return torch.cat((A1,B1), dim=1)

JITTER = 1e-6

# a = torch.tensor()
# %%
x1 = torch.linspace(1,6,6).reshape(-1,1)
x2 = torch.linspace(1,5,5).reshape(-1,1)

x_vec = combinations(x1,x2)

# C = kronecker(A,B)
# D = combinations(A,B)
# A1 = A.unsqueeze(-1).expand(A.size(0), A.size(1), B.size(0))
# A1 =A.unsqueeze(0).expand(B.size(0),A.size(0), A.size(1)).reshape(B.size(0)*A.size(0), A.size(1))
# 
# A1 = A.repeat(B.size(0), 1)
# B1 = B.unsqueeze(0).expand(A.size(0),B.size(0), B.size(1)).reshape(A.size(0)*B.size(0), B.size(1))
# B1 = A.expand(A.size(0)*B.size(0), A.size(1))


true_function = lambda x:  x.sum(dim=1).pow(2)

y_vec = true_function(x_vec)

plt.plot(y_vec.data)

y_matix = y_vec.view(6,5)
# %%
from stgp_v01 import stgp
model = stgp(x1, x2, y_matix)

y2 = model(x1, x2)
plt.plot(y2.data)
# %%
