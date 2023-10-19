import lmdb
import os
import cv2
from tqdm import tqdm
import numpy as np
import json
'''NOTE
Type data: 
    1. a image folder and json label
    2. a image folder and a text folder 
'''

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True

def create_lmdb_data(args):
    # Đường dẫn đến folder chứa các file ảnh và folder label
    if args.raw_data_type=='json':
        json_labels = os.path.join(args.own_data_path, "labels.json")
        with open(json_labels, 'r') as f:
            data = json.load(f)
        # ratio = 0.5
        os.makedirs(os.path.join(args.lmdb_data_path, args.data_mode), exist_ok=True)
        env = lmdb.open(os.path.join(args.lmdb_data_path, args.data_mode), map_size=int(1e12))
        with env.begin(write=True) as txn:
            # Duyệt qua các file ảnh trong folder
            cnt = 1            
            for image_file, label in tqdm(data.items()):
                # Đọc file ảnh
                try:
                    with open(os.path.join(args.own_data_path, 'data', image_file), 'rb') as f:
                        imageBin = f.read()
                except:
                    with open(os.path.join(args.own_data_path, image_file), 'rb') as f:
                        imageBin = f.read()


                # Ghi dữ liệu vào LMDB
                imageKey = 'image-%09d'.encode() % cnt
                labelKey = 'label-%09d'.encode() % cnt
                txn.put(imageKey, imageBin)
                txn.put(labelKey, label.encode())
                cnt += 1
            txn.put('num-samples'.encode(), str(cnt-1).encode())
        # Đóng file LMDB
        env.close()
    
    if args.raw_data_type=='folder':
 
        os.makedirs(os.path.join(args.lmdb_data_path, args.data_mode), exist_ok=True)

        image_folder = os.path.join(args.raw_data_path, 'images')
        label_folder = os.path.join(args.raw_data_path, f'{args.data_mode}_annotations')

        env = lmdb.open(os.path.join(args.lmdb_data_path, args.data_mode), map_size=int(1e12))
        with env.begin(write=True) as txn:
            cnt = 1
            for labels in os.listdir(label_folder):
                with open(os.path.join(label_folder, labels), "r") as l:
                    l_data = l.readlines()
                    for line in l_data:
                        image_name, text = line.split("\t")

                        with open(os.path.join(image_folder, image_name), 'rb') as f:
                            imageBin = f.read()
                        if 'synth_data' not in image_folder:
                            nparr = np.fromstring(imageBin, np.uint8)
                            img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            h, w, c = img_cv2.shape
                            if h < 5 or w < 5:
                                # print('Continue')
                                continue

                        # Đọc label tương ứng với file ảnh
                        # label_file = image_file[:-4] + ".txt"
                        # with open(os.path.join(label_folder, label_file), "r") as f:
                        #     label = f.read().strip()
                        label = text

                        # Ghi dữ liệu vào LMDB
                        imageKey = 'image-%09d'.encode() % cnt
                        labelKey = 'label-%09d'.encode() % cnt
                        txn.put(imageKey, imageBin)
                        txn.put(labelKey, label.encode())
                        cnt += 1

                    txn.put('num-samples'.encode(), str(cnt-1).encode())
        env.close()