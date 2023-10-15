from .loss import CTCLoss
from .metric import RecMetric
from .postprocess.rec_postprocess import CTCLabelDecode

from .pipeline import SVTRArch, PPLCNetV3Arch

def get_models(args):
    if args.model=='SVTR':
        model = SVTRArch()
    
    elif args.model=='LCNETV3':
        model = PPLCNetV3Arch()

    return model

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