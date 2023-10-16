
import dataloader
from nnet.postprocess.rec_postprocess import BaseRecLabelDecode, CTCLabelDecode

# ctc_decoder = BaseRecLabelDecode('./utils/vi_dict.txt', True)
ctc_decoder = CTCLabelDecode('./utils/vi_dict.txt', True)
from dataloader.data.imaug.label_ops import BaseRecLabelEncode

base_encode = BaseRecLabelEncode(60, './utils/vi_dict.txt', True)

def test_batch_data(args):
    train_loader, eval_loader = dataloader.get_loader(args)
    for batch1, batch2 in zip(train_loader, eval_loader):
        print(batch2['image'].shape)
        print(batch2['label'][0].shape)
        print((batch2['label'].numpy().tolist()[0]))
        
        print(ctc_decoder.idx_to_char(batch2['label'].numpy().tolist()[0]))
        break