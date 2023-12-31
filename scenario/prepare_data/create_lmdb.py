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
    
    elif args.raw_data_type=='folder':
 
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
    # my hard code
    else: 
        # # Đường dẫn đến folder chứa các file ảnh và folder label
        # # 40k 50k 60k 50k 100k 30k
        # train_paths = [
        #     # {'image_folder': './data/pretrain/images_hw1',
        #     # 'label_folder': './data/pretrain/labels_hw1'},
        #     # {'image_folder': './data/pretrain/images_hw2',
        #     # 'label_folder': './data/pretrain/labels_hw2'},
        #     # {'image_folder': './data/pretrain/images_hw3',
        #     # 'label_folder': './data/pretrain/labels_hw3'},
        #     # {'image_folder': './data/pretrain/images_hw4',
        #     # 'label_folder': './data/pretrain/labels_hw4'},
        #     # {'image_folder': './data/pretrain/images_hw_normal',
        #     # 'label_folder': './data/pretrain/labels_hw_normal'}
        #     # {'image_folder': './data/kalapa_fix/train/images',
        #     # 'label_folder': './data/kalapa_fix/train/labels'},
        #     {'image_folder': './data/train_finetune_0311_v2_split_img/train/images',
        #     'label_folder': './data/train_finetune_0311_v2_split_img/train/labels'},
        # ]

        # val_paths = [
        #     {'image_folder': './data/train_finetune_0311_v2_split_img/val/images',
        #     'label_folder': './data/train_finetune_0311_v2_split_img/val/labels'}]
        # paths = train_paths
        # paths = val_paths
        # Đường dẫn đến file LMDB sẽ được tạo
        # lmdb_path = "./ocr_reg_lmdb/train/"
        train_data_path = os.path.join(args.raw_data_path, "train")
        val_data_path = os.path.join(args.raw_data_path, "val")

        lmdb_val_path = os.path.join(args.lmdb_data_path, "val")
        lmdb_train_path = os.path.join(args.lmdb_data_path, "train")

        os.makedirs(lmdb_val_path, exist_ok=True)
        os.makedirs(lmdb_train_path, exist_ok=True)
        # Mở file LMDB để ghi dữ liệu

        train_env = lmdb.open(lmdb_train_path, map_size=int(1e12))

        # Khởi tạo transaction để ghi dữ liệu
        with train_env.begin(write=True) as txn:
            # Duyệt qua các file ảnh trong folder
            cnt = 1
            cache = {}
            for path in os.listdir(train_data_path):
                print(path)
                if path.startswith("images"):
                    image_folder = os.path.join(train_data_path, path)
                    label_folder = os.path.join(train_data_path, path.replace("images", "labels"))
    
                    for image_file in tqdm(sorted(os.listdir(image_folder))):
                        # Đọc file ảnh
                        with open(os.path.join(image_folder, image_file), 'rb') as f:
                            imageBin = f.read()
                        if 'synth_data' not in image_folder:
                            nparr = np.fromstring(imageBin, np.uint8)
                            img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            h, w, c = img_cv2.shape
                            if h < 5 or w < 5:
                                # print('Continue')
                                continue
                        # Đọc label tương ứng với file ảnh
                        label_file = image_file[:-4] + ".txt"
                        with open(os.path.join(label_folder, label_file), "r") as f:
                            label = f.read().strip()
                            if len(label)>= args.max_text_length: 
                                print(label_file)
                                continue
                        # Ghi dữ liệu vào LMDB
                        imageKey = 'image-%09d'.encode() % cnt
                        labelKey = 'label-%09d'.encode() % cnt
                        txn.put(imageKey, imageBin)
                        txn.put(labelKey, label.encode())
                        cnt += 1

                txn.put('num-samples'.encode(), str(cnt-1).encode())

        # Đóng file LMDB
        train_env.close()


        val_env = lmdb.open(lmdb_val_path, map_size=int(1e12))

        # Khởi tạo transaction để ghi dữ liệu
        with val_env.begin(write=True) as txn:
            # Duyệt qua các file ảnh trong folder
            cnt = 1
            cache = {}
            for path in os.listdir(val_data_path):
                print(path)
                if path.startswith("images"):
                    image_folder = os.path.join(val_data_path, path)
                    label_folder = os.path.join(val_data_path, path.replace("images", "labels"))
    
                    for image_file in tqdm(sorted(os.listdir(image_folder))):
                        # Đọc file ảnh
                        with open(os.path.join(image_folder, image_file), 'rb') as f:
                            imageBin = f.read()
                        if 'synth_data' not in image_folder:
                            nparr = np.fromstring(imageBin, np.uint8)
                            img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            h, w, c = img_cv2.shape
                            if h < 5 or w < 5:
                                # print('Continue')
                                continue
                        # Đọc label tương ứng với file ảnh
                        label_file = image_file[:-4] + ".txt"
                        with open(os.path.join(label_folder, label_file), "r") as f:
                            label = f.read().strip()
                            
                        imageKey = 'image-%09d'.encode() % cnt
                        labelKey = 'label-%09d'.encode() % cnt
                        txn.put(imageKey, imageBin)
                        txn.put(labelKey, label.encode())
                        cnt += 1

                txn.put('num-samples'.encode(), str(cnt-1).encode())

        # Đóng file LMDB
        val_env.close()


        