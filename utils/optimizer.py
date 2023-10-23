import torch

def get_optimizer(model, args):
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    elif args.optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01)    
        
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        raise NotImplementedError
    return optimizer


def get_lr_scheduler(optimizer, args):
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.T_max, eta_min=0)
    return lr_scheduler