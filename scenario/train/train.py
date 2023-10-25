from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import torch
import tqdm
import os

from torch.nn import CTCLoss
import utils
from .util import get_ckpt_folder, get_logs_folder, get_ckpt_name, get_pretrain_folder

import nnet
import dataloader

class Trainer:
    def __init__(self, args):
        self.args = args
        self.train_loader, self.eval_loader = dataloader.get_loader(args)
        self.model = nnet.get_models(args).to(self.args.device)
        nnet.print_summary(self.model, verbose=True)

        self.postprocess = nnet.get_postprocess(args)

        self.optimizer = utils.get_optimizer(self.model, args)

        self.lr_scheduler = utils.get_lr_scheduler(self.optimizer, args)

        self.log_folder = get_logs_folder(args)
        self.writer = SummaryWriter(self.log_folder)

        self.metric = nnet.get_metric(args)

        self.criterion = CTCLoss(zero_infinity=True)

        self.iteration = 0

        

    def train_step(self, batch, batch_idx):
        image, label, label_length= batch['image'], batch['label'], batch['length']
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
            # print("verify shape: {}".format(output.shape))
            
            postprocessed_output = self.postprocess(output.cpu().detach().numpy(),
                                                    label.cpu().numpy())
            print(postprocessed_output)
            metric_ = self.metric(postprocessed_output)

        return metric_
    

    def _fit(self):
        self.load_checkpoint()

        for epoch in range(self.args.num_epoch):
            # eval 
            self.model.eval()
            with tqdm.tqdm(self.eval_loader, unit="it") as pbar:
                pbar.set_description(f'Evaluate epoch {epoch}')
                test_accuracy = []
                test_norm_edit_dis = []
                test_cer = []
                for batch_idx, batch in enumerate(pbar):
                    # validate
                    metric = self.eval_step(batch, batch_idx)
                    test_accuracy.append(float(metric['acc']))
                    test_norm_edit_dis.append(float(metric['norm_edit_dis']))
                    test_cer.append(float(metric['cer_score']))
                    pbar.set_postfix(accuracy=float(metric['acc']))
            self.write_eval_metric_to_tensorboard(epoch, metric)
            
            # train
            self.model.train()
            torch.autograd.set_detect_anomaly(True)
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
            # if self.accuracy < np.mean(test_accuracy):
            if (self.norm_edit_dis < np.mean(test_norm_edit_dis)) and (self.accuracy < np.mean(test_accuracy)):
                self.accuracy = np.mean(test_accuracy)
                self.norm_edit_dis = np.mean(test_norm_edit_dis)
                self.cer = np.mean(test_cer)
                self.save_checkpoint()

            if self.lr_scheduler != None:
                self.lr_scheduler.step(self.accuracy)
        
    def fit(self):
        try:
            self._fit()
        except KeyboardInterrupt:
            print('fuck model')


    def clip_grad_norm(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
    
    def write_eval_metric_to_tensorboard(self, epoch, metrics):
        accuracy = metrics['acc']
        norm_edit_dis = metrics['norm_edit_dis']
        cer = metrics['cer_score']
        with open(f'{os.path.join(self.log_folder, "train_log.txt")}', 'a') as fin:
            fin.write(f'Evaluate epoch {epoch} - acc: {accuracy} - norm_edit_dis: {norm_edit_dis} - cer: {cer}\n')
        print(f'Evaluate epoch {epoch} - acc: {accuracy} - norm_edit_dis: {norm_edit_dis} - cer: {cer}\n')

        # write to tensorboard
        self.writer.add_scalars('validation metric', metrics, epoch)

    def write_train_metric_to_tensorboard(self, loss):
        loss_dict = {"loss": float(loss)}
        self.writer.add_scalars('training metric', loss_dict, self.iteration)



    def get_checkpoint_path(self):
        ckpt_folder = get_ckpt_folder(self.args)
        ckpt_name = get_ckpt_name(self.args)
        return os.path.join(ckpt_folder, ckpt_name) + '.ckpt'
    
    def get_pretrain_path(self):
        ckpt_folder = get_pretrain_folder(self.args)
        ckpt_name = get_ckpt_name(self.args)
        return os.path.join(ckpt_folder, ckpt_name) + '.ckpt'
    
    def load_checkpoint(self):
        self.epoch = 0
        self.accuracy = 0.
        self.cer = 0.
        self.norm_edit_dis = 0.
        path = self.get_pretrain_path()
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.epoch = checkpoint['epoch']
            self.iteration = checkpoint['iteration']
            # try:
            #     self.accuracy = checkpoint['accuracy']
            #     self.self.cer = checkpoint['cer']
            #     self.norm_edit_dis = checkpoint['norm_edit_dis']
            # except:
            #     self.accuracy = 0.
            #     self.cer = 0.
            #     self.norm_edit_dis = 0.
            print(f'Best accuracy: {self.accuracy}')

    
    def save_checkpoint(self):
        # save checkpoint
        torch.save({
            'accuracy': self.accuracy,
            'cer': self.cer,
            'norm_edit_dis': self.norm_edit_dis,
            'iteration': self.iteration,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            }, self.get_checkpoint_path())
        print('[+] checkpoint saved')

    
def train(args):
    trainer = Trainer(args)
    trainer.fit()
