import os
import csv
import glob
import torch
import nnet
import time
from utils.util import *
from utils.args import get_args
from utils.preprocess import detect_text_lines
import shutil
import time


def edit_distance(s1, s2):
    m=len(s1)+1
    n=len(s2)+1
    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    return tbl[i,j]

def Convert_list(string):
    # Remove newline characters from the string
    cleaned_string = string.replace('\n', '')
    cleaned_string = cleaned_string.replace(' ', '')
    # Convert the cleaned string to a list of characters
    list1 = []
    list1[:0] = cleaned_string
    return list1

if __name__ == "__main__":

    folder_in = "./Dataset/kalapa_valid_fixed/images"
    anno_folder_in = "./Dataset/kalapa_valid_fixed/labels"
    folder_checker = "./Dataset/checker"

    args = get_args()
    print(args.model)

    model = nnet.get_models(args)
    model = model.to(args.device)
    ckpt_path = "./CP/SVTR_2410.ckpt"
    # ckpt_path = "./ckpt/SVTR_best_200epochs.ckpt"
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    print(model)
    postprocess = nnet.get_postprocess(args)

    imgC, imgH, imgW = (3,48,720)
    max_wh_ratio = imgW / imgH

    CER = 0
    acc_true = 0
    total_img = 0
    t_start = time.time()
    for path in tqdm(os.listdir(folder_in)):
        image = cv2.imread(os.path.join(folder_in, path))
        # preprocess image input
        image = detect_text_lines(image)
        
        path_anno = os.path.join(anno_folder_in, os.path.basename(path)[:-4] + '.txt')
        with open(path_anno, 'r') as f:
            for line in f:
                GT = line

        total_img += 1
        norm_img = resize_norm_img(image, max_wh_ratio)
        norm_img = torch.tensor(np.expand_dims(norm_img, axis=0)).to(args.device)
        output = model(norm_img)[0]
        predict = postprocess(output.cpu().detach().numpy())[0][0]


        result_text = Convert_list(predict)
        label_text_gt = Convert_list(GT)

        cer = edit_distance(result_text, label_text_gt) / len(label_text_gt)
        CER+=cer

        if result_text == label_text_gt:
            acc_true += 1
        else:
            img_save = f'{str(cer)}=gt={GT}=pre={predict}.jpg'
            cv2.imwrite(os.path.join(folder_checker , img_save), image)

    print("Total image process = ", total_img)
    print('Time per img = ', (time.time() - t_start)/total_img )
    print('Acc = ', 100 * acc_true/total_img, " %" )
    print("CER = ", CER/total_img)
