from .dataset import LMDBDataSet

from torch.utils.data import DataLoader


def get_loader(args):

    def collate_fn(batch):
        pass

    train_dataset = LMDBDataSet(args, mode='train')
    train_loader = DataLoader(dataset=train_dataset,
                            drop_last=False,
                            batch_size=args.batch_size,
                            num_workers=args.num_worker)
    
    eval_dataset = LMDBDataSet(args, mode='eval')
    eval_loader = DataLoader(dataset=eval_dataset,
                             drop_last=False,
                             batch_size=args.batch_size,
                             num_workers=args.num_worker)
    
    return train_loader, eval_loader
