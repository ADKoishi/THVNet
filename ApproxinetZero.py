import torch
from torch import nn

from Utils.layerGenerator import getTransformLayer, getActivationLayer
from setTransformer.models import SetTransformer


class ApproximaNetZero(nn.Module):
    def __init__(
            self, transInputDim, transNHead=4, hiddenDim=128, transOutputDim=128, transOutputNum=1,
            useSAB=False, ISABInds=16,
            forwardLayers=8, resOn=False, batchNorm=False, dropOut=0.0
    ):
        super(ApproximaNetZero, self).__init__()

        self.forwardLayers = forwardLayers
        self.resOn = resOn

        # if self.forwardLayers % 2 != 0 and self.resOn:
        #     self.resOn = False
        #     print("ResNet enabled but with odd forward layers, automatically disabled ResNet forward")

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
        for i in range(forwardLayers):
            self.decoderLayers.append(
                getTransformLayer(
                    forwardDim=transOutputDim,
                    batchNorm=batchNorm
                )
            )
            self.decoderLayers.append(
                getActivationLayer(dropOut=dropOut)
            )

        # If this is vacant here,
        # Loop in the Forward will fetch layer into cpu and terminate
        # the computation
        self.decoderLayers = nn.ModuleList(self.decoderLayers)

        # Last layer, for HV aggregation
        self.aggregationLayer = nn.Sequential(
            nn.Linear(transOutputDim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch_size, element_num, input_dim)

        out = self.encoder(x)
        # out: (batch_size, num_outputs, dim_output)

        out = torch.sum(out, dim=1)
        # out: (batch_size, dim_output)

        former_out = out.clone()

        for i in range(len(self.decoderLayers)):
            if self.resOn and i % 4 == 0:
                former_out = out.clone()

            if self.resOn and i % 4 == 3:
                out = out + former_out

            out = self.decoderLayers[i](out)
        # out: (batch_size, dim_output)

        out = self.aggregationLayer(out)
        # out: (batch_size, 1)

        out = torch.squeeze(out)
        # out: (batch_size)

        return out
