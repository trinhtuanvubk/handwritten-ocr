from .loss import CTCLoss
from .metric import RecMetric
from .postprocess.rec_postprocess import CTCLabelDecode, BaseRecLabelDecode

from .pipeline import SVTRArch, PPLCNetV3Arch

from .loss import CTCLoss

from .ngram.decoder import vi_dict, BeamCTCDecoder

def get_models(args):
    # try:
    if args.model=='SVTR':
        model = SVTRArch()
    
    elif args.model=='LCNETV3':
        model = PPLCNetV3Arch()
    
    return model
    # except:
    #     print("Only support SVTR or LCNETV3")

def get_loss(args):
    ctc_loss = CTCLoss(args)
    return ctc_loss  

def get_postprocess(args):
    ctc_label_decode = CTCLabelDecode(args.character_dict_path, args.use_space_char)
    return ctc_label_decode

def get_metric(args):
    metric = RecMetric()
    return metric

def print_summary(model, verbose=False):
    if verbose:
        print(model)
    print('Trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Non-trainable parameters:', sum(p.numel() for p in model.parameters() if not p.requires_grad))