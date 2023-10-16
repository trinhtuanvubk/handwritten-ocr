from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import torch
import tqdm
import os


import utils
from .util import *

import nnet
import dataloader

class Trainer:
    def __init__(self, args):
        self.args = args
        self.train_loader, self.eval_loader = dataloader.get_loader(args)
        self.model = nnet.get_models(args)
        nnet.print_summary(self.model, verbose=True)

        self.optimizer = utils.get_optimizer(args)
        self.lr_scheduler = utils.get_lr_scheduler(self.optimizer, args)

        self.writer = SummaryWriter(get_logs_folder(args))

        self.metric = nnet.get_metric(args)

        self.iteration = 0

    def train_step(self, batch, batch_idx):
        image, label = batch['image'], batch['label']
        image = image.to(self.args.device)
        label = label.to(self.args.device)

        self.optimizer.zero_grad()

        output = self.model(image)

        # loss = 

