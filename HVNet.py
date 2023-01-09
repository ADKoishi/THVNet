import torch
from torch import nn


class HVNet(nn.Module):
    def __init__(self, hidden_dim=256, input_dim=5, encoder_layers=4, decoder_layers=4, res_on=True):
        super(HVNet, self).__init__()
        self.res_on = res_on
        self.encoder_block = []
        self.encoder_block.append(nn.Linear(input_dim, hidden_dim))
        for idx in range(encoder_layers):
            self.encoder_block.append(nn.ReLU())
            self.encoder_block.append(nn.Linear(hidden_dim, hidden_dim))
        self.decoder_block = []
        for idx in range(decoder_layers):
            self.decoder_block.append(nn.Linear(hidden_dim, hidden_dim))
            self.decoder_block.append(nn.ReLU())
        self.aggregation_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.encoder_layers = nn.ModuleList(self.encoder_block)
        self.decoder_layers = nn.ModuleList(self.decoder_block)
    def forward(self, X):
        for encoder_layer in self.encoder_layers:
            X = encoder_layer(X)
        X = torch.sum(X, dim=1)
        former_out = None
        for idx in range(len(self.decoder_layers)):
            if self.res_on and idx % 4 == 0:
                former_out = X.clone()
            if self.res_on and idx % 4 == 3:
                X = X + former_out
            X = self.decoder_layers[idx](X)
        X = torch.squeeze(self.aggregation_layer(X))
        return X

