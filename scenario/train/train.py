from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import torch
import tqdm
import os

from torch.nn import CTCLoss
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

        self.optimizer = utils.get_optimizer(self.model, args)
        self.lr_scheduler = utils.get_lr_scheduler(self.optimizer, args)

        self.writer = SummaryWriter(get_logs_folder(args))

        self.metric = nnet.get_metric(args)

        self.criterion = CTCLoss()

        self.iteration = 0

    def train_step(self, batch, batch_idx):
        image, label, label_length= batch['image'], batch['label']
        image = image.to(self.args.device)
        label = label.to(self.args.device)

        self.optimizer.zero_grad()

        output = self.model(image)

        # permute to compute loss (should be fixed soon)
        permuted_output = output.permute(1,0,2)
        N, B, _ = permuted_output.shape
        output_length = torch.tensor([N]*B, dtype=torch.int64)

        loss_dict = {"loss": self.criterion(permuted_output, label, output_length, label_length)}
        
        # DEBUG
        if torch.isnan(loss_dict['loss']):
            print('shit nan', loss_dict['loss'])
            raise KeyboardInterrupt            
        # END DEBUG
        # backward and update weight
        loss_dict['loss'].backward()

        self.clip_grad_norm()
        self.optimizer.step()

        return loss_dict

    def clip_grad_norm(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
    
    def write_eval_metric_to_tensorboard(self, epoch, metrics):
        # compute average
        for key in metrics:
            metrics[key] = np.mean(metrics[key])
        # display
        print('Evaluate epoch:{}: si_snr={:0.2f} pesq={:0.2f},  stoi={:0.2f}, estoi={:0.2f}' \
            .format(epoch, metrics['acc'], metrics['norm_edit_dis']))
        # write to tensorboard
        self.writer.add_scalars('validation metric', metrics, epoch)

    def write_train_metric_to_tensorboard(self, loss_dicts):
        for key in loss_dicts:
            loss_dicts[key] = np.mean(loss_dicts[key])
        self.writer.add_scalars('training metric', loss_dicts, self.iteration)
    

    

