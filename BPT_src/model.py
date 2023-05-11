import os
import sys

import torch
import torch.nn as nn
import numpy as np
from pretrain.pre_model import _weights_init
from scipy import signal
from torch.autograd import Function
from workspace.pretrain_graph.config.gene_cfg import BasicCfg


class GRL(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


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

    def forward(self, z, alpha=1, reverse_gradient=False):
        if reverse_gradient:
            z = GRL.apply(z, alpha)
        logit = self.mlp(z)

        return logit
