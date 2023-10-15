
import dataloader
def test_batch_data(args):
    train_loader, eval_loader = dataloader.get_loader(args)
    for batch in train_loader:
        print(batch)