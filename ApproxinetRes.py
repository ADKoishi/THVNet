import torch
from torch import nn

from Utils.layerGenerator import getTransformLayer, getActivationLayer
from setTransformer.models import SetTransformer


class ApproximaNetRes(nn.Module):
    def __init__(
            self, transInputDim, transNHead=4, hiddenDim=128, transOutputDim=128, transOutputNum=1,
            useSAB=False, ISABInds=32,
            fullForwardLayers=8, halfForwardLayers=8, quarterForwardLayers=8, eightForwardLayers=8,
            resOn=False, batchNorm=False, dropOut=0.0
    ):
        super(ApproximaNetRes, self).__init__()

        self.fullForwardDim = transOutputDim * transOutputNum
        self.fullForwardLayers = fullForwardLayers

        self.halfForwardDim = self.fullForwardDim // 2
        self.halfForwardLayers = halfForwardLayers

        self.quarterForwardDim = self.halfForwardDim // 2
        self.quarterForwardLayers = quarterForwardLayers

        self.eighthForwardDim = self.quarterForwardDim // 2
        self.eighthForwardLayers = eightForwardLayers

        self.resOn = resOn

        if self.resOn \
                and self.fullForwardLayers % 2 != 0 \
                or self.halfForwardDim % 2 != 0 \
                or self.quarterForwardDim % 2 != 0 \
                or self.eighthForwardDim % 2 != 0:

            self.resOn = False
            print("ResNet enabled but with odd forward layers, automatically disabled ResNet forward")

        self.encoder = SetTransformer(
            dim_input=transInputDim,
            num_outputs=transOutputNum,
            dim_output=transOutputDim,
            use_sab=useSAB,
            num_inds=ISABInds,
            dim_hidden=hiddenDim,
            num_heads=transNHead
        )

        self.decoderLayers = []
        self.blockInfo = [
            (fullForwardLayers, self.fullForwardDim),
            (halfForwardLayers, self.halfForwardDim),
            (quarterForwardLayers, self.quarterForwardDim),
            (eightForwardLayers, self.eighthForwardDim),
        ]

        for layerInfo in self.blockInfo:
            for i in range(layerInfo[0]):
                self.decoderLayers.append(
                    getTransformLayer(
                        forwardDim=layerInfo[1],
                        batchNorm=batchNorm
                    )
                )
                self.decoderLayers.append(
                    getActivationLayer(dropOut=dropOut)
                )
            # Dimension reduction layer
            self.decoderLayers.append(
                nn.Linear(layerInfo[1], layerInfo[1] // 2)
            )

        # Remove the last dimension reduction layer
        self.decoderLayers.pop(-1)

        # If this is vacant here,
        # Loop in the Forward will fetch layer into cpu and terminate
        # the computation
        self.decoderLayers = nn.ModuleList(self.decoderLayers)

        # Last layer, for HV aggregation
        self.aggregationLayer = nn.Sequential(
            nn.Linear(self.eighthForwardDim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch_size, element_num, input_dim)

        out = self.encoder(x)
        # out: (batch_size, num_outputs, dim_output)

        out = torch.flatten(out, start_dim=1, end_dim=2)
        # out: (batch_size, dim_output*num_outputs)

        accumulate_layer = 0
        former_out = out.clone()
        for layerInfo in self.blockInfo:
            for i in range(layerInfo[0] * 2):
                if self.resOn and i % 4 == 0:
                    former_out = out.clone()

                if self.resOn and i % 4 == 3:
                    out = out + former_out
                out = self.decoderLayers[accumulate_layer](out)
                accumulate_layer += 1

            # Dimension reduction layer
            if accumulate_layer < len(self.decoderLayers):
                out = self.decoderLayers[accumulate_layer](out)
                accumulate_layer += 1

        # out: (batch_size, eighthForwardDim)

        out = self.aggregationLayer(out)
        # out: (batch_size, 1)

        out = torch.squeeze(out)
        # out: (batch_size)

        return out
