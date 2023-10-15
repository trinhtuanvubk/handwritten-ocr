import os

def get_ckpt_folder(args):
    folder = f'../ckpt/{args.model}/checkpoints'
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def get_ckpt_name(args):
    return '{}'.format(args.model)

def get_logs_folder(args):
    return os.path.join(get_ckpt_folder(args).replace('checkpoints', 'logs'), get_ckpt_name(args))
