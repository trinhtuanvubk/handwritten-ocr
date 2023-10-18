from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import torch
import tqdm
import os

from torch.nn import CTCLoss
import utils
from .util import get_ckpt_folder, get_logs_folder, get_ckpt_name

import nnet
import dataloader

class Trainer:
    def __init__(self, args):
        self.args = args
        self.train_loader, self.eval_loader = dataloader.get_loader(args)
        self.model = nnet.get_models(args)
        nnet.print_summary(self.model, verbose=True)

        self.postprocess = nnet.get_postprocess(args)

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
        output = output[0]

        # permute to compute loss (should be fixed soon)
        permuted_output = output.permute(1,0,2)
        N, B, _ = permuted_output.shape
        output_length = torch.tensor([N]*B, dtype=torch.int64)

        loss = self.criterion(permuted_output, label, output_length, label_length)
        
        # DEBUG
        if torch.isnan(loss):
            print('shit nan', loss)
            raise KeyboardInterrupt            
        # END DEBUG
        # backward and update weight
        loss.backward()

        self.clip_grad_norm()
        self.optimizer.step()

        return loss
    
    def eval_step(self, batch, batch_idx):
        with torch.no_grad():
            image, label = batch['image'], batch['label']
            image = image.to(self.args.device)
            label = label.to(self.args.device)

            output = self.model(image)
            output = output[0]
            
            postprocessed_output = self.postprocess(output.cpu().detach().numpy(),
                                                    label.cpu().numpy())
            
            metric_ = self.metric(postprocessed_output)

        return metric_
    

    def _fit(self):
        self.load_checkpoit()

        for epoch in range(self.args.num_epoch):
            # eval 
            self.model.eval()
            with tqdm.tqdm(self.eval_loader, unit="it") as pbar:
                pbar.set_description(f'Evaluate epoch {epoch}')
                test_accuracy = []
                for batch_idx, batch in enumerate(pbar):
                    # validate
                    metric = self.eval_step(batch, batch_idx)
                    test_accuracy.append(float(metric['acc']))
                    pbar.set_postfix(accuracy=float(metric['acc']))
            self.write_eval_metric_to_tensorboard(epoch, np.mean(test_accuracy))
            
            # train
            self.model.train()
            with tqdm.tqdm(self.train_loader, unit="it") as pbar:
                pbar.set_description(f'Epoch {epoch}')
                for batch_idx, batch in enumerate(pbar):

                    # perform training step
                    loss = self.train_step(batch, batch_idx)
                    pbar.set_postfix(loss=float(loss))

                    # log
                    self.epoch = epoch
                    self.iteration += 1
                    if self.iteration % self.args.log_iter == 0:
                        self.write_train_metric_to_tensorboard(loss)

            # save checkpoint
            if self.accuracy < np.mean(test_accuracy):
                self.accuracy = np.mean(test_accuracy)
                self.save_checkpoint()

            if self.scheduler != None:
                self.scheduler.step(self.accuracy)
        
    def fit(self):
        try:
            self._fit()
        except KeyboardInterrupt:
            print('fuck model')


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

    def write_train_metric_to_tensorboard(self, loss):
        loss_dict = {"loss": float(loss)}
        self.writer.add_scalars('training metric', loss_dict, self.iteration)



    def get_checkpoint_path(self):
        ckpt_folder = get_ckpt_folder(self.args)
        ckpt_name = get_ckpt_name(self.args)
        return os.path.join(ckpt_folder, ckpt_name) + '.ckpt'
    
    def load_checkpoit(self):
        self.epoch = 0
        self.accuracy = 0.
        path = self.get_checkpoint_path()
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.epoch = checkpoint['epoch']
            self.iteration = checkpoint['iteration']
            self.accuracy = checkpoint['accuracy']
            print(f'Best accuracy: {self.accuracy}')

    
    

    
def train(args):
    trainer = Trainer(args)
    trainer.fit()
