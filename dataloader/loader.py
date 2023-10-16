import torch
from .dataset import LMDBDataSet, load_config

from torch.utils.data import DataLoader

def collate_fn(batch):
    return {
        'image': torch.stack([torch.tensor(x['image']) for x in batch]),
        'label': torch.stack([torch.tensor(x['label']) for x in batch])
}

def get_loader(args):


    config = load_config(args.pre_config_path)
    print(config)

    train_dataset = LMDBDataSet(args, config, mode='train')
    train_loader = DataLoader(dataset=train_dataset,
                            drop_last=False,
                            collate_fn=collate_fn,
                            batch_size=args.batch_size,
                            num_workers=args.num_worker)
    
    eval_dataset = LMDBDataSet(args, config, mode='eval')
    eval_loader = DataLoader(dataset=eval_dataset,
                             drop_last=False,
                             collate_fn=collate_fn,
                             batch_size=args.batch_size,
                             num_workers=args.num_worker)
    
    return train_loader, eval_loader
