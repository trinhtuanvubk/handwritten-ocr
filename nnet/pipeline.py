import torch
import torch.nn as nn

from .svtrnet import SVTRNet
from .lcnetv3 import PPLCNetV3

from .modules.encoder import SequenceEncoder
from .modules.rec_head import CTCHead

class SVTRArch(nn.Module):
    def __init__(self):
        super(SVTRArch, self).__init__()
        self.backbone = SVTRNet()
        self.neck = SequenceEncoder(in_channels=192, encoder_type='reshape')
        self.head = CTCHead(in_channels=192)
    
    def forward(self, x):
        x = self.backbone(x)
        # print(x.shape)
        x = self.neck(x)
        # print(x.shape)
        x = self.head(x)
        # print("outshape model: {}".format(x[0].shape))
        return x
    
class PPLCNetV3Arch(nn.Module):
    def __init__(self):
        super(PPLCNetV3Arch, self).__init__()
        self.backbone = PPLCNetV3()
        self.neck = SequenceEncoder(in_channels=512, encoder_type='svtr')
        self.head = CTCHead(in_channels=192)
    
    def forward(self, x):
        x = self.backbone(x)
        # print(x.shape)
        x = self.neck(x)
        # print(x.shape)
        x = self.head(x)
        # print("outshape model: {}".format(x[0].shape))
        return x
    


