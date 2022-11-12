import numpy as np
import scipy.io as scio

# dataFile = "./3D.mat"
# sets = scio.loadmat(dataFile)
# dataSet = np.array(sets['Data'][0])
# dataSet = np.nan_to_num(dataSet)
# print(dataSet)
import torch
from torch import nn

test = np.array([[1,2,3],[2,3,4]])

print(test.transpose((1,0)))