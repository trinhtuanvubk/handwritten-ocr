
import torch
from torch import nn


'''NOTE
not using now
'''
class CTCLoss(nn.Module):
    def __init__(self, use_focal_loss=False, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')
        self.use_focal_loss = use_focal_loss

    def forward(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        predicts = predicts.permute(1, 0, 2)
        N, B, _ = predicts.shape
        preds_lengths = torch.tensor(
            [N] * B, dtype=torch.int64)
        labels = batch[1].astype("int32")
        label_lengths = batch[2].astype('int64')
        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        if self.use_focal_loss:
            weight = torch.exp(-loss)
            weight = torch.subtract(torch.tensor([1.0]), weight)
            weight = torch.square(weight)
            loss = torch.multiply(loss, weight)
        loss = loss.mean()
        return {'loss': loss}
