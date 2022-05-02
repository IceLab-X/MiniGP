import torch
torch.set_default_tensor_type(torch.DoubleTensor)
import torch.nn as nn
import numpy as np
import math
import tensorly
from matplotlib import pyplot as plt
from scipy.io import loadmat

import os
from cigp_v10 import cigp
from cigp_v10_rho import ConstMeanCIGP
from sklearn.metrics import mean_squared_error, r2_score

print("cigp_lar:", torch.__version__)
# I use torch (1.11.0) for this work. lower version may not work.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Fixing strange error if run in MacOS
JITTER = 1e-6
EPS = 1e-10
PI = 3.1415

# calculate r2 rmse
def calculate_metrix(**kwargs):
    """
        calculates r2, rmse and mnll of model prediction.
        kwargs:
        :param y_test: ndarray or tensor
        :param y_mean_pre: ndarray or tensor
        :param y_var_pre: ndarray or tensor
    """
    # check if arguments is ndarray type
    for key, arg in kwargs.items():
        if type(arg) is torch.Tensor:
            kwargs[key] = kwargs[key].detach().numpy()
    # R2
    r2 = r2_score(kwargs['y_test'], kwargs['y_mean_pre'])
    # RMSE
    rmse = np.sqrt(mean_squared_error(kwargs['y_test'], kwargs['y_mean_pre']))
    # Test log likelihood
    # mnll = -np.sum(scipy.stats.norm.logpdf(kwargs['y_test'],
    #                                        loc=kwargs['y_mean_pre'],
    #                                        scale=np.sqrt(kwargs['y_var_pre']))) / len(kwargs['y_test'])
    return {'r2': r2, 'rmse': rmse}

def main(data_file):
    # load data
    data = loadmat(data_file)

    # prep data for cigp
    n_train = 50
    n_test = 28
    train_y = data['Ytr_interp'][0].tolist()[0][0:n_train]
    train_yh = data['Ytr_interp'][0].tolist()[2][0:n_train][::4]
    xtr_l = data['xtr'][0:50]
    xtr_h = data['xtr'][0:50][::4]

    eval_y = data['Yte_interp'][0].tolist()[0][0:n_test]
    eval_yh = data['Yte_interp'][0].tolist()[2][0:n_test]
    eval_x = data['xte'][0:n_test]

    xtr_l = torch.stack([torch.from_numpy(xtr_l[i]) for i in range(50)],dim = 0)
    xtr_h = torch.stack([torch.from_numpy(xtr_h[i]) for i in range(13)],dim = 0)
    ytr_l = torch.stack([torch.from_numpy(tensorly.tensor_to_vec(train_y[i])) for i in range(50)], dim = 0)
    ytr_h = torch.stack([torch.from_numpy(tensorly.tensor_to_vec(train_yh[i])) for i in range(13)], dim = 0)

    xte_l = torch.stack([torch.from_numpy(eval_x[i]) for i in range(28)],dim = 0)
    yte_l = torch.stack([torch.from_numpy(tensorly.tensor_to_vec(eval_y[i])) for i in range(28)], dim = 0)
    yte_h = torch.stack([torch.from_numpy(tensorly.tensor_to_vec(eval_yh[i])) for i in range(28)], dim = 0)

    iter_num  = 100

    '''train low-fidelity GP'''
    model_l = cigp(xtr_l, ytr_l)
    model_l.train_adam(iter_num, lr=0.01)
    with torch.no_grad():
        ypred_l, yvar_l = model_l(xte_l)
        metrics_LF = calculate_metrix(y_test=yte_l, y_mean_pre=ypred_l)
    print("loss of low-fidelity GP:",metrics_LF)


    '''train high_lar GP'''
    # preparation for high fidelity GP training
    with torch.no_grad():
        ytr_m, ytr_v = model_l(xtr_h)

    yln = ytr_l[0].numel()
    yhn = ytr_h[0].numel()
    # 初始化参数：训练用x， 训练用y， 低精度数据的size（289 = 17*17）， 高精度数据size
    model_LAR = ConstMeanCIGP(xtr_h, ytr_h, yln, yhn)
    model_LAR.train_adam(ytr_m, 100, lr = 0.01)

    '''predict and test high_lar GP'''
    with torch.no_grad():
        yte_m, yte_v = model_l(xte_l)

    with torch.no_grad():
        ypred_lar= model_LAR(xte_l, ytr_m, ytr_v, yte_m, yte_v)

    metrics_lar = calculate_metrix(y_test=yte_h,
                                    y_mean_pre=ypred_lar)
    print("loss of LAR:", metrics_lar)
    print("loss of low-fidelity:", metrics_LF)

if __name__ == '__main__':
    data_file = "/Users/aaaalison/Desktop/hogpp/Hogp/MultiFidelity_ReadyData/poisson_v4_02.mat"
    main(data_file)