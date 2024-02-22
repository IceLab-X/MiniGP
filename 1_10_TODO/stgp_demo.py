# demo file for stgp

# %%

import scipy
import numpy as np
from scipy.io import loadmat

# import data from a sample data saved in matlab .mat
data = loadmat('H:\\eda\\MF主仓库\\Mini-GP\\1_10_TODO\\data_v1_09_05_fill.mat')     #sample data saved in matlab .mat
# data = loadmat('data_v1_00_fill.mat')     #sample data saved in matlab .mat


time = data['time'] #time for the pm25 measurement data pm25

str = data['sTr']   #spatial coordinate [lat,long,elevation] for all sensors in SLC
pm25 = data['pm25'] #pm25 observed values for all sensors

str_daq = data['sTr_DAQ'] #spatial coordinate for daq sensors
pm25_daq = data['pm25_daq'] #reading from daq sensors


# import stgp_v01_2
from stgp_v02 import *
# model = stgp(str, time, pm25, noise_variance = 0.1)

str = torch.tensor(str)     #convert data to pytorch tensor
time = torch.tensor(time)   #convert data to pytorch tensor
pm25 = torch.tensor(pm25)   #convert data to pytorch tensor

# model = stgp(str, time, pm25, noise_variance = 0.1)

model = stgp(str, time, pm25,
             latlong_length_scale=4300.,
             elevation_length_scale=30.,
             time_length_scale=0.25,
             noise_variance=0.1)

str_daq = torch.tensor(str_daq)
yPred, yVar = model(str_daq, time)

plt.plot(yPred.T,'--')  #show the predictions at the daq locations
plt.plot(pm25_daq.T,'-')    #show the daq observations
plt.show()

# %%
model.train_adam(5,0.1)    #optimize hyperparameter using adam optimizer
# model.train_bfgs(1,0.0001)     #optimize hyperparameter using l-BFGS-B

yPred, yVar = model(str_daq, time)      #make predictions at the daq locations

plt.plot(yPred.T,'--')  #show the predictions at the daq locations
plt.plot(pm25_daq.T,'-')     #show the daq observations
plt.show()

