import numpy as np
import torch
from torch.utils.data import DataLoader

from ApproxinetZero import ApproximaNetZero

from Utils.HVDataLoader import HVDataset

import matplotlib.pyplot as plt



# Model parameters
TARGET_NUM = 5
TRANS_OUT_NUM = 16
TRANS_OUT_DIM = 120
HIDDEN_DIM = 120
DROP_OUT = 0
USE_SAB = True
USE_RES = True
USE_BATCH_NORM = True
COSINE_ANNEALING = True
ACTIVATION = "ReLU"

# Training parameters for Normal version
FORWARD_LAYERS = 8

# Training parameters for Res version
LAYER_DEPTH = 6

# Training parameters
LUCKY_SEED = 114514
NUM_EPOCH = 80
TRAIN_PROPORTION = 0.9
LEARNING_RATE = 1e-5
BATCH_SIZE = 200
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    print("Device Found, Named", DEVICE)
    approximator = ApproximaNetZero(
        transInputDim=TARGET_NUM,
        transNHead=TARGET_NUM,
        transOutputNum=TRANS_OUT_NUM,
        transOutputDim=TRANS_OUT_DIM,
        hiddenDim=HIDDEN_DIM,
        useSAB=USE_SAB,
        forwardLayers=FORWARD_LAYERS,
        resOn=USE_RES,
        batchNorm=USE_BATCH_NORM,
        dropOut=DROP_OUT
    ).to(DEVICE)
    testSet = HVDataset(dataDir="./Datasets/Short", objectNum=TARGET_NUM, seeds=[5])
    testLoader = DataLoader(
        dataset=testSet,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=6,
        pin_memory=True,
    )
    optimizer = torch.optim.SGD(
        approximator.parameters(),
        lr=LEARNING_RATE,
        momentum=0.95,
        weight_decay=1e-5
    )
    print(f"test loader len: {len(testLoader)}")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=15*LEARNING_RATE,
        total_steps=NUM_EPOCH*len(testLoader),
        pct_start=0.25,
        cycle_momentum=True,
        three_phase=True
    )

    steps = []
    lrs = []
    cnt = 0
    for epoch in range(NUM_EPOCH):
        for step in range(len(testLoader)):
            steps.append(cnt)
            scheduler.step()
            cnt += 1
            lrs.append(scheduler.get_last_lr())
    print(np.array(lrs)[0:-1:len(testLoader)])
    plt.figure()
    plt.plot(steps,lrs)
    plt.show()

