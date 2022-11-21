import random
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ApproxinetRes import ApproximaNetRes
from ApproxinetZero import ApproximaNetZero

from Loss import MLSEloss, PCTLoss

from Utils.HVDataLoader import getDataLoader, HVDataset

from tqdm import tqdm


# Function to ensure experiments replicable
def lock_random(luckySeed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(luckySeed)
    torch.manual_seed(luckySeed)
    random.seed(luckySeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(luckySeed)
        torch.cuda.manual_seed_all(luckySeed)


# Model parameters
TARGET_NUM = 5
TRANS_OUT_NUM = 16
TRANS_OUT_DIM = 240
HIDDEN_DIM = 240
DROP_OUT = 0
USE_SAB = True
USE_RES = True
USE_BATCH_NORM = True
COSINE_ANNEALING = True
ACTIVATION = "ReLU"

# Training parameters for Normal version
FORWARD_LAYERS = 8

# Training parameters for Res version
LAYER_DEPTH = 4

# Training parameters
LUCKY_SEED = 114514
NUM_EPOCH = 254
TRAIN_PROPORTION = 0.9
LEARNING_RATE = 1e-5
BATCH_SIZE = 200
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

MODEL = 0

RESCUE_START = 61
if __name__ == "__main__":
    lock_random(LUCKY_SEED)

    trainDataLoader, validDataLoader = getDataLoader(
        batchSize=BATCH_SIZE,
        workerNum=4,
        objectNum=TARGET_NUM,
        trainProportion=TRAIN_PROPORTION,
        seeds=[3, 4]
    )

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

    model_name = "./models/Zero_TransOut16_TARGET_NUM5_TRANS_OUT_NUM16_TRANS_OUT_DIM240_HIDDEN_DIM240_ACTIVATIONReLU_FC8_ResTrue_BNTrue.ckpt"
    approximator.load_state_dict(torch.load(model_name))
    criterion = MLSEloss()
    optimizer = torch.optim.AdamW(approximator.parameters(), lr=15*LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=2, T_mult=2
    )

    result_name = model_name + '.rst.txt'
    with open(result_name, mode='r+') as result_file:
        result_txt = result_file.read()
        min_valid_loss = float(re.findall("Loss: .*", result_txt)[-1].split(": ")[-1])

    for epoch in range(RESCUE_START):
        scheduler.step()
    with open(result_name, mode='a+') as result_file:
        result_file.write(f"\nResume Training, scheduler recovered. Latest learning rate: {scheduler.get_last_lr()}\n")
    print(f"Scheduler recovered! Latest learning rate: {scheduler.get_last_lr()}")

    # Main training loop
    for epoch in range(RESCUE_START, NUM_EPOCH):
        # Training
        approximator.train()
        pbar = tqdm(trainDataLoader)
        accumulate_loss = 0
        batch_num = 0
        for batch in pbar:
            batch_num += 1
            VS, HV = batch

            results = approximator(VS.to(DEVICE))
            loss = criterion(results, HV.to(DEVICE))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accumulate_loss += loss.item()
            pbar.set_postfix(
                now_epoch=epoch,
                loss=loss.item(),
                average_epoch_loss=accumulate_loss / batch_num,
                mode="training",
                learning_rate=scheduler.get_last_lr()
            )
        with open(result_name, mode='a+') as result_file:
            result_file.write(
                f'now_epoch={epoch}, '
                f'loss={loss.item()}, '
                f'average_epoch_loss={accumulate_loss / batch_num}, '
                f'mode=training, '
                f'learning_rate={scheduler.get_last_lr()}\n'
            )
        if COSINE_ANNEALING:
            scheduler.step()

        # Validation
        approximator.eval()
        pbar = tqdm(validDataLoader)
        accumulate_loss = 0
        batch_num = 0
        for batch in pbar:
            batch_num += 1
            VS, HV = batch

            results = approximator(VS.to(DEVICE))
            with torch.no_grad():
                loss = criterion(results, HV.to(DEVICE))

            accumulate_loss += loss.item()
            pbar.set_postfix(
                now_epoch=epoch,
                loss=loss.item(),
                average_epoch_loss=accumulate_loss / batch_num,
                mode="validation"
            )
        with open(result_name, mode='a+') as result_file:
            result_file.write(
                f'now_epoch={epoch}, '
                f'loss={loss.item()}, '
                f'average_epoch_loss={accumulate_loss / batch_num}, '
                f'mode=validation\n'
            )

        if accumulate_loss / batch_num < min_valid_loss:
            min_valid_loss = accumulate_loss / batch_num

            torch.save(
                approximator.state_dict(),
                model_name
            )
            print(f"Model in Epoch {epoch} Saved! With Average Validation Loss: {accumulate_loss / batch_num}")
            with open(result_name, mode='a+') as result_file:
                result_file.write(
                    f"Model in Epoch {epoch} Saved! With Average Validation Loss: {accumulate_loss / batch_num}\n"
                )

            time.sleep(0.1)

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
    with open(result_name, mode='a+') as result_file:
        result_file.write(str(accumulate_loss / batch_num) + "\n")
