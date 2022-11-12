import torch
import torch.nn as nn


class MLSEloss(nn.Module):
    def __init__(self):
        super(MLSEloss, self).__init__()
        self.MSE = nn.MSELoss()

    def forward(self, output, target):
        return self.MSE(torch.log(output), torch.log(target))


class PCTLoss(nn.Module):
    def __init__(self):
        super(PCTLoss, self).__init__()

    def forward(self, output, target):
        return torch.mean(
            torch.divide(
                torch.abs(
                    torch.sub(target, output)
                ),
                target
            )
        )
