import torch
import torch.nn as nn
from pretrain.pre_model import _weights_init

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            # nn.BatchNorm1d(in_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim//2, in_dim//4),
            # nn.BatchNorm1d(in_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim//4, out_dim),
        )
        self.apply(_weights_init)

    def forward(self, z):
        logit = self.mlp(z)
        return logit
