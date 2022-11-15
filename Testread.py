import re

import numpy as np
import scipy.io as scio

# dataFile = "./3D.mat"
# sets = scio.loadmat(dataFile)
# dataSet = np.array(sets['Data'][0])
# dataSet = np.nan_to_num(dataSet)
# print(dataSet)
import torch
from torch import nn