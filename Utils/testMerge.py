import h5py
import numpy as np

data = h5py.File("merged_data.mat", 'r')
rawDataSet = np.array(data['Data'])
rawHV = np.array(data['HVval'])
print(rawDataSet.shape)
print(rawHV.shape)