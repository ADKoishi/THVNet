import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ApproxinetRes import ApproximaNetRes
from ApproxinetZero import ApproximaNetZero
from HVNet import HVNet

from Loss import PCTLoss

from Utils.HVDataLoader import HVDataset

from tqdm import tqdm

# Model parameters
TARGET_NUM = 5
TRANS_OUT_NUM = 16
TRANS_OUT_DIM = 64
HIDDEN_DIM = 64
USE_SAB = False
USE_RES = True
USE_BATCH_NORM = False
COSINE_ANNEALING = True

# Training parameters for Normal version
FORWARD_LAYERS = 4

# Training parameters for Res version
LAYER_DEPTH = 6

# Training parameters
LUCKY_SEED = 114514
BATCH_SIZE = 100
DEVICE = 'cpu'
MODEL = 0

if __name__ == "__main__":
    print("Device Found, Named", DEVICE)
    if MODEL == 0:
        approximator = ApproximaNetZero(
            transInputDim=TARGET_NUM,
            transOutputNum=TRANS_OUT_NUM,
            transOutputDim=TRANS_OUT_DIM,
            hiddenDim=HIDDEN_DIM,
            useSAB=USE_SAB,
            forwardLayers=FORWARD_LAYERS,
            resOn=USE_RES,
            batchNorm=USE_BATCH_NORM,
        ).to(DEVICE)
    if MODEL == 1:
        approximator = HVNet(
            hidden_dim=HIDDEN_DIM,
            input_dim=TARGET_NUM,
            encoder_layers=FORWARD_LAYERS,
            decoder_layers=FORWARD_LAYERS,
            res_on=USE_RES
        ).to(DEVICE)


    model_name = "models/Zero_USE_SABFalse_TARGET_NUM5_TRANS_OUT_NUM16_TRANS_OUT_DIM64_HIDDEN_DIM64_FC4_ResTrue_BNFalse.ckpt"

    # Prediction
    testSet = HVDataset(dataDir="./Datasets/Short", objectNum=TARGET_NUM, seeds=[5])

    testLoader = DataLoader(
        dataset=testSet,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=6,
        pin_memory=True,
    )

    approximator.load_state_dict(torch.load(model_name, map_location='cuda:0'))
    approximator.eval()

    pbar = tqdm(testLoader)
    criterion = PCTLoss()
    accumulate_loss = 0
    batch_num = 0
    start_time = time.time()
    for batch in pbar:
        batch_num += 1
        VS, HV = batch
        with torch.no_grad():
            results = approximator(VS.to(DEVICE))
        loss = criterion(results, HV.to(DEVICE))

        accumulate_loss += loss.item()

        pbar.set_postfix(
            loss=loss.item(),
            average_epoch_loss=accumulate_loss / batch_num,
            mode="testing"
        )
    print(time.time() - start_time)
