import argparse
import torch
import os

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--scenario', type=str, default='test_output_model')
    parser.add_argument('--scenario', type=str, default='train')
# 

    parser.add_argument('--model', type=str, default='SVTR')

    parser.add_argument('--raw_data_path', type=str, default="./data/OCR/training_data")
    parser.add_argument('--raw_data_type', type=str, default='folder', help='types: json or folder')
    parser.add_argument('--lmdb_data_path', type=str, default='./data/kalapa_lmdb/')
    # parser.add_argument('--ratio_lmdb', type=float, default=0.8, help="train test split")
    parser.add_argument('--data_mode', type=str, default='train', help="to create folder train or eval")
    parser.add_argument('--pre_config_path', type=str, default='./dataloader/config.yml')
    parser.add_argument('--character_dict_path', type=str, default='./utils/vi_dict.txt')
    parser.add_argument('--use_space_char', action='store_false')
    # parser.add_argument('--train_path', type=str, default='./data/own_lmdb/train/')
    # parser.add_argument('--eval_path', type=str, default='./data/own_lmdb/eval/')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--T_max', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epoch', type=str, default=2000)
    parser.add_argument('--log_iter', type=int, default=10)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--shuffle', action='store_false')
    args = parser.parse_args()

    args.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    return args