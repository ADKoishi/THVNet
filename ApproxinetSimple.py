import torch
from torch import nn

from Utils.layerGenerator import getTransformLayer, getActivationLayer
from setTransformer.models import SetTransformer


class ApproximaNetZero(nn.Module):
    def __init__(
            self, transInputDim, transNHead=4, hiddenDim=128,
            useSAB=False, ISABInds=32,
            forwardLayers=8, resOn=False, batchNorm=False, dropOut=0.0
    ):
        super(ApproximaNetZero, self).__init__()

        self.forwardLayers = forwardLayers
        self.resOn = resOn

        if self.forwardLayers % 2 != 0 and self.resOn:
            self.resOn = False
            print("ResNet enabled but with odd forward layers, automatically disabled ResNet forward")

        self.encoder = SetTransformer(
            dim_input=transInputDim,
            num_outputs=1,
            dim_output=1,
            use_sab=useSAB,
            num_inds=ISABInds,
            dim_hidden=hiddenDim,
            num_heads=transNHead
        )

    def forward(self, x):
        # x: (batch_size, element_num, input_dim)

        out = self.encoder(x)
        # out: (batch_size, num_outputs, dim_output)

        return out.item()
