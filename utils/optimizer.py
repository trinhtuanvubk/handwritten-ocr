import torch

def get_optimizer(model, args):
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        raise NotImplementedError
    return optimizer

