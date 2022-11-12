# Function to get a "Transform Layer"
# with a linear layer and a possible batchNorm layer included
# for the model
import torch
from torch import nn

from setTransformer.models import SetTransformer


def getTransformLayer(forwardDim, batchNorm):
    transformLayer = nn.Sequential(
        nn.Linear(forwardDim, forwardDim)
    )

    if batchNorm:
        transformLayer.append(
            nn.BatchNorm1d(forwardDim)
        )

    return transformLayer


# Function to get a "Activation Layer"
# with a activation function and a possible dropOut layer included
# for the model
def getActivationLayer(dropOut):
    activationLayer = nn.Sequential(
        nn.ReLU()
    )

    if dropOut > 0:
        activationLayer.append(
            nn.Dropout(dropOut)
        )

    return activationLayer