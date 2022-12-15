import numpy as np
import torch
import torch.nn as nn
from torch import optim


import warnings

warnings.filterwarnings("ignore")


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _select_optimizer(self):
        # sourcery skip: inline-immediately-returned-variable
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        pass
