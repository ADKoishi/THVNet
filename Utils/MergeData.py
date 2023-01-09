import h5py
import numpy as np
import torch

TARGET_LENGTH = 10


def mergeData(dataDir, extractSolutionSize):
    objectives = (5,)
    seeds = (3, 4)
    mergedTensor = torch.zeros((0, 100, TARGET_LENGTH))
    mergedHV = torch.zeros(0)
    for objective in objectives:
        for seed in seeds:
            dataPath = f"{dataDir}/train_data_M{objective}_{seed}.mat"
            data = h5py.File(dataPath, 'r')
            rawDataSet = np.array(data['Data']).transpose((2, 1, 0))
            rawHV = np.array(data['HVval']).transpose((1, 0))
            selectedIndices = np.random.choice(rawDataSet.shape[0], size=extractSolutionSize, replace=False)
            selectedDataSet = rawDataSet[selectedIndices]
            selectedHV = torch.squeeze(
                torch.FloatTensor(
                    rawHV[selectedIndices]
                )
            )
            paddedTensor = torch.FloatTensor(
                np.pad(
                    selectedDataSet,
                    ((0, 0), (0, 0), (0, TARGET_LENGTH - objective)),
                    constant_values=((0, 0), (0, 0), (0, 0))
                )
            )
            mergedTensor = torch.cat([mergedTensor, paddedTensor.clone()])
            mergedHV = torch.cat([mergedHV, selectedHV.clone()])
    mergedTensor = torch.nan_to_num(mergedTensor)
    with h5py.File('../Datasets/merged_data.mat', 'w') as result:
        result.create_dataset('Data', data=mergedTensor)
        result.create_dataset('HVval', data=mergedHV)

if __name__ == "__main__":
    mergeData("../Datasets/Short", 1000)
