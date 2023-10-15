import argparse
import torch
import os

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--scenario', type=str, default='train_model')

    parser.add_argument('--own_data_path', type=str, default="./data/data_sample_eval/")
    parser.add_argument('--raw_data_type', type=str, default='json', help='types: json or folder')
    parser.add_argument('--lmdb_data_path', type=str, default='./data/own_lmdb/')

    parser.add_argument('--character_dict_path', type=str, default='./utils/vi_dict.txt')
    parser.add_argument('--use_space_char', action=False)
    # parser.add_argument('--train_path', type=str, default='./data/own_lmdb/train/')
    # parser.add_argument('--eval_path', type=str, default='./data/own_lmdb/eval/')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_worker', type=int, default=4)
    args = parser.parse_args()

    args.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    return args