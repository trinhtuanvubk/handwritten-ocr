
import dataloader
from nnet.postprocess.rec_postprocess import BaseRecLabelDecode, CTCLabelDecode
# from nnet import get_models, get_postprocess, get_metric

import nnet
import utils

args = utils.get_args()
args.model = 'LCNETV3'
ctc_decoder = CTCLabelDecode('./utils/vi_dict.txt', True)

def test_output_model(args):
    train_loader, eval_loader = dataloader.get_loader(args)
    model = nnet.get_models(args)
    loss = nnet.get_loss(args)
    postprocess = nnet.get_postprocess(args)
    for batch1, batch2 in zip(train_loader, eval_loader):
        image, label = batch2['image'], batch2['label']
        image = image.to(args.device)
        label = label.to(args.device)
        print(image.shape)
        print(label.shape)

        model_output = model(image)
        print(model_output[0].shape)
        # model_output = model_output[0].permute(1, 0, 2)
        model_output = model_output[0]
        
        output = postprocess(model_output.cpu().detach().numpy())
        print(len(output))
        print(output)

        loss_ = loss(model_output, label)
        print(loss_)

        break
        