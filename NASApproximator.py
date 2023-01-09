import copy
import datetime
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from torch.utils.data import DataLoader

from ApproxinetZero import ApproximaNetZero
from HVNet import HVNet

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
USE_SAB = False
USE_RES = True
USE_BATCH_NORM = False
COSINE_ANNEALING = True

# Training parameters for Normal version

# Training parameters
LUCKY_SEED = 114514
NUM_EPOCH = 1
TRAIN_PROPORTION = 0.9
LEARNING_RATE = 1e-5
BATCH_SIZE = 200

# NSGA-II parameters
GEN = 2
POP_SIZE = 2
DEVICES = [0]
CPU = "cpu"


class MyProblem(Problem):
    def __init__(self, **kwargs):
        self.trainDataLoader, self.validDataLoader = getDataLoader(
            batchSize=BATCH_SIZE,
            workerNum=4,
            dataDir="./Datasets/NAS",
            objectNum=TARGET_NUM,
            trainProportion=TRAIN_PROPORTION,
            seeds=[3, 4]
        )
        super().__init__(**kwargs)

    def _evaluate(self, population, out, *args, **kwargs):
        res = []
        individual_cnt = 0
        for individual in population:
            individual_cnt += 1
            print(f"Now processing individual count: {individual_cnt}")
            trans_out_num = [1, 4, 8, 12, 16][individual[0]]
            trans_out_dim = individual[1] * 8
            hidden_dim = individual[2] * 8
            forward_layers = individual[3]
            model = ApproximaNetZero(
                transInputDim=TARGET_NUM,
                transOutputNum=trans_out_num,
                transOutputDim=trans_out_dim,
                hiddenDim=hidden_dim,
                useSAB=USE_SAB,
                forwardLayers=forward_layers,
                resOn=USE_RES,
                batchNorm=USE_BATCH_NORM
            )
            model = torch.nn.DataParallel(model, device_ids=DEVICES)
            model = model.cuda(device=DEVICES[0])
            criterion = MLSEloss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=15 * LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=2, T_mult=2
            )
            # Main training loop
            min_valid_loss = torch.inf
            best_dict = None
            for epoch in tqdm(range(NUM_EPOCH)):
                model.train()
                accumulate_loss = 0
                batch_num = 0
                for batch in self.trainDataLoader:
                    batch_num += 1
                    VS, HV = batch
                    VS, HV = VS.cuda(device=DEVICES[0]), HV.cuda(device=DEVICES[0])
                    result = model(VS)
                    loss = criterion(result, HV)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    accumulate_loss += loss.item()
                if COSINE_ANNEALING:
                    scheduler.step()
                # Validation
                model.eval()
                accumulate_loss = 0
                batch_num = 0
                for batch in self.validDataLoader:
                    batch_num += 1
                    VS, HV = batch
                    VS, HV = VS.cuda(device=DEVICES[0]), HV.cuda(device=DEVICES[0])
                    result = model(VS)
                    with torch.no_grad():
                        loss = criterion(result, HV)
                    accumulate_loss += loss.item()
                if accumulate_loss / batch_num < min_valid_loss:
                    min_valid_loss = accumulate_loss / batch_num
                    best_dict = copy.deepcopy(model.state_dict())

            testSet = HVDataset(dataDir="./Datasets", objectNum=TARGET_NUM, seeds=[5])
            testLoader = DataLoader(
                dataset=testSet,
                batch_size=BATCH_SIZE,
                shuffle=False,
                drop_last=True,
                num_workers=6,
                pin_memory=True,
            )
            model.load_state_dict(best_dict)
            model.eval()
            criterion = PCTLoss()
            accumulate_loss = 0
            batch_num = 0
            start_time = time.time()
            for batch in testLoader:
                batch_num += 1
                VS, HV = batch
                VS, HV = VS.cuda(device=DEVICES[0]), HV.cuda(device=DEVICES[0])
                with torch.no_grad():
                    result = model(VS)
                loss = criterion(result, HV)
                accumulate_loss += loss.item()
            time_cost = time.time() - start_time
            avg_pct_loss = accumulate_loss / batch_num
            res.append([avg_pct_loss, time_cost])
            del model, criterion, optimizer, scheduler

        out['F'] = np.array(res)


problem = MyProblem(n_var=4, n_obj=2, xl=[0, 1, 1, 0], xu=[4, 32, 32, 4], vtype=int)

if __name__ == "__main__":
    lock_random(LUCKY_SEED)

    print(f"Enabled cuda devices: {DEVICES}")
    # [
    #   trans_out_num 1, 4, 8, 12, 16
    #   trans_out_dim 8~256: (1~32) mult 8,
    #   hidden_dim 8~256: (1~32) mult 8,
    #   forward_layers: 0~4: 2bit
    # ]
    algorithm = NSGA2(
        pop_size=POP_SIZE,
        sampling=IntegerRandomSampling(),
        crossover=SBX(eta=15, prob=0.9, repair=RoundingRepair()),
        mutation=PM(eta=20, repair=RoundingRepair()),
        eliminate_duplicates=True
    )
    stop_criteria = ('n_gen', GEN)
    results = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=stop_criteria,
        save_history=True
    )
    timestamp = datetime.datetime.now().timestamp()
    Scatter().add(results.F).save(f"{timestamp}.F.png")
    torch.save({'X': results.X, 'F': results.F}, f"NAS_results_{timestamp}")
