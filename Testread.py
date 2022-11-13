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

with open("models/Zero_TransOut16_TARGET_NUM5_TRANS_OUT_NUM16_TRANS_OUT_DIM128_HIDDEN_DIM125_ACTIVATIONReLU_FC16_ResTrue_BNTrue.ckpt.rst.txt", mode='r+') as result_file:
    file_txt = result_file.read()
    print(float(re.findall("Loss: .*", file_txt)[-1].split(": ")[-1]))