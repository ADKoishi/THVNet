import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from ApproxinetRes import ApproximaNetRes
from ApproxinetZero import ApproximaNetZero

from Loss import PCTLoss

from Utils.HVDataLoader import HVDataset

from tqdm import tqdm

# Model parameters
TARGET_NUM = 5
TRANS_OUT_NUM = 16
TRANS_OUT_DIM = 128
HIDDEN_DIM = (128 // TARGET_NUM) * TARGET_NUM
DROP_OUT = 0
USE_SAB = True
USE_RES = True
USE_BATCH_NORM = True
COSINE_ANNEALING = True
ACTIVATION = "ReLU"

# Training parameters for Normal version
FORWARD_LAYERS = 32

# Training parameters for Res version
LAYER_DEPTH = 6

# Training parameters
LUCKY_SEED = 114514
NUM_EPOCH = 125
TRAIN_PROPORTION = 0.9
LEARNING_RATE = 1e-5
BATCH_SIZE = 200
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

MODEL = 0

if __name__ == "__main__":
    print("Device Found, Named", DEVICE)
    if MODEL == 0:
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
    else:
        approximator = ApproximaNetRes(
            transInputDim=TARGET_NUM,
            transNHead=TARGET_NUM,
            transOutputNum=TRANS_OUT_NUM,
            transOutputDim=TRANS_OUT_DIM,
            hiddenDim=HIDDEN_DIM,
            useSAB=USE_SAB,
            fullForwardLayers=LAYER_DEPTH,
            halfForwardLayers=LAYER_DEPTH,
            quarterForwardLayers=LAYER_DEPTH,
            eightForwardLayers=LAYER_DEPTH,
            resOn=USE_RES,
            batchNorm=USE_BATCH_NORM,
            dropOut=DROP_OUT
        ).to(DEVICE)

    model_name = \
        f"./models/" \
        f"{'Zero' if MODEL == 0 else 'Res'}_" \
        f"TransOut{TRANS_OUT_NUM}_" \
        f"TARGET_NUM{TARGET_NUM}_"\
        f"TRANS_OUT_NUM{TRANS_OUT_NUM}_"\
        f"TRANS_OUT_DIM{TRANS_OUT_DIM}_"\
        f"HIDDEN_DIM{HIDDEN_DIM}_"\
        f"ACTIVATION{ACTIVATION}_"\
        f"FC{FORWARD_LAYERS}_" \
        f"Res{USE_RES}_" \
        f"BN{USE_BATCH_NORM}.ckpt"

    # Prediction
    testSet = HVDataset(dataDir="./Datasets", objectNum=TARGET_NUM, seeds=[5])

    testLoader = DataLoader(
        dataset=testSet,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=6,
        pin_memory=True,
    )

    approximator.load_state_dict(torch.load(model_name))
    approximator.eval()

    pbar = tqdm(testLoader)
    criterion = PCTLoss()
    accumulate_loss = 0
    batch_num = 0
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
