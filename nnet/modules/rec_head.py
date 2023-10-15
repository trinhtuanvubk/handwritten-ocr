import math

import torch
import torch.nn as nn
from torch.nn import functional as F

class CTCHead(nn.Module):
    def __init__(self,
                 in_channels=192,
                 out_channels=125,
                 fc_decay=0.0004,
                 mid_channels=None,
                 return_feats=False,
                 **kwargs):
        super(CTCHead, self).__init__()
        if mid_channels is None:
            self.fc = nn.Linear(
                in_channels,
                out_channels)
        else:
            self.fc1 = nn.Linear(
                in_channels,
                mid_channels)

            self.fc2 = nn.Linear(
                mid_channels,
                out_channels)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            stdv = 1.0 / math.sqrt(self.in_channels * 1.0)
            nn.init.uniform_(m.weight, -stdv, stdv)
            nn.init.uniform_(m.bias, -stdv, stdv)

    def forward(self, x):
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)
        #   batchsize * T * C ---->  T * batchsize * C
        predicts = predicts.permute(1, 0, 2)
        predicts = predicts.log_softmax(2).requires_grad_()

        if self.return_feats:
            result = (predicts, x)
        else:
            result = (predicts, None)
        return result
