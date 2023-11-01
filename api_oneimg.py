import os
import csv
import glob
import torch
import nnet
import time
from utils.util import *
from utils.args import get_args
import shutil

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

    img_path = "/home/sonlt373/Desktop/SoNg/OCR_handwriting_shop_sticker/dev/handwritten-ocr/0.07142857142857142=gt=Quận 8 Hồ Chí Minh=pre=Quận 3 Hồ Chí Minh.jpg"
    image = cv2.imread(img_path)

    norm_img = resize_norm_img(image, max_wh_ratio)
    norm_img = torch.tensor(np.expand_dims(norm_img, axis=0)).to(args.device)
    output = model(norm_img)[0]
    predict = postprocess(output.cpu().detach().numpy())[0][0]

    print("[result==]", predict)
