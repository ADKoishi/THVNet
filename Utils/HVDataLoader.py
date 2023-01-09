import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, random_split, DataLoader


class HVDataset(Dataset):
    def __init__(self, objectNum, dataDir="./Datasets", seeds=(3, 4, 5)):
        self.dadaFiles = [
            dataDir +
            "/train_data_M" +
            str(objectNum) +
            "_" + str(seed) +
            ".mat" for seed in seeds
        ]

        self.data = []
        self.labels = []
        for dataFile in self.dadaFiles:
            singleSeedData = h5py.File(dataFile, 'r')
            self.data.append(
                torch.nan_to_num(
                    torch.FloatTensor(
                        np.array(singleSeedData['Data']).transpose((2, 1, 0))
                    )
                )
            )
            self.labels.append(
                torch.FloatTensor(
                    np.array(singleSeedData['HVval']).transpose(1, 0)
                )
            )
        self.data = torch.concat(self.data, dim=0)
        self.labels = torch.squeeze(torch.concat(self.labels, dim=0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

class MeregedDataset(Dataset):
    def __init__(self, path):
        datafile = h5py.File(path, 'r')
        self.data = torch.FloatTensor(np.array(datafile['Data']))
        self.labels = torch.FloatTensor(np.array(datafile['HVval']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

def getDataLoader(
        batchSize,
        workerNum,
        objectNum,
        trainProportion=0.9,
        dataDir="./Datasets",
        seeds=(3, 4, 5),
        merge=False,
        mergedPath = "./Datasets/merged_data.mat"
):
    if not merge:
        fullDataset = HVDataset(dataDir=dataDir, objectNum=objectNum, seeds=seeds)
    else:
        fullDataset = MeregedDataset(path=mergedPath)
    trainLen = int(trainProportion * len(fullDataset))
    validLen = len(fullDataset) - trainLen
    trainSet, validSet = random_split(fullDataset, [trainLen, validLen])

    trainLoader = DataLoader(
        dataset=trainSet,
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
        num_workers=workerNum,
        pin_memory=True,
    )
    validLoader = DataLoader(
        dataset=validSet,
        batch_size=batchSize,
        shuffle=True,
        drop_last=True,
        num_workers=workerNum,
        pin_memory=True,
    )
    return trainLoader, validLoader

