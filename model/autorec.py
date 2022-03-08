import torch
from torch import nn
import pytorch_lightning as pl

class AutoRec(nn.Module):

    def __init__(self,
                input_size: int,
                hidden_size: int):
        #super(AutoRec, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


