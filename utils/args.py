import argparse
import torch
import os

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--scenario', type=str, default='train_model')

    parser.add_argument('--own_data_path', type=str, default="./data/data_sample_eval/")
    parser.add_argument('--raw_data_type', type=str, default='json', help='types: json or folder')
    parser.add_argument('--lmdb_data_path', type=str, default='./data/own_lmdb/eval')


    args = parser.parse_args()

    args.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    return args