
import dataloader
from nnet.postprocess.rec_postprocess import BaseRecLabelDecode, CTCLabelDecode
# from nnet import get_models, get_postprocess, get_metric
import torch
import nnet
import utils
from torch.nn import CTCLoss
args = utils.get_args()
args.model = 'LCNETV3'
ctc_decoder = CTCLabelDecode('./utils/vi_dict.txt', True)
criterion = CTCLoss()
metric = nnet.get_metric(args)
def test_output_model(args):
    train_loader, eval_loader = dataloader.get_loader(args)
    model = nnet.get_models(args)
    loss = nnet.get_loss(args)
    postprocess = nnet.get_postprocess(args)
    for batch1, batch2 in zip(train_loader, eval_loader):
        image, label, label_length = batch2['image'], batch2['label'], batch2['length']
        image = image.to(args.device)
        label = label.to(args.device)
        
        print(image.shape)
        print(label.shape)

        model_output = model(image)
        print(model_output[0].shape)
        # model_output = model_output[0].permute(1, 0, 2)
        model_output = model_output[0]
        
        # output = postprocess(model_output.cpu().detach().numpy())
        # print(len(output))
        # print(output)
        model_output_ = model_output.permute(1,0,2)
        N, B, _ = model_output_.shape
        output_length = torch.tensor([N]*B, dtype=torch.int64)

        loss_ = criterion(model_output_, label, output_length, label_length)
        print(f"loss: {loss_}")

        output = postprocess(model_output.cpu().detach().numpy(), label.cpu().numpy())
        print(output)
        metric_ = metric(output)
        print(f"metric: {metric_}")
        # metric: {'acc': 0.0, 'norm_edit_dis': 4.99997500014171e-06}

        break
        