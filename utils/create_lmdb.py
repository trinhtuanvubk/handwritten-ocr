import lmdb
import os
import cv2
from tqdm import tqdm
import numpy as np

# Đường dẫn đến folder chứa các file ảnh và folder label
train_paths = [
    {'image_folder': './new_bgr_reg_datasets/train/images20/',
     'label_folder': './new_bgr_reg_datasets/train/labels20/'},
     {'image_folder': './new_bgr_reg_datasets/train/images21/',
     'label_folder': './new_bgr_reg_datasets/train/labels21/'},
     {'image_folder': './new_bgr_reg_datasets/train/images22/',
     'label_folder': './new_bgr_reg_datasets/train/labels22/'},
    #   {'image_folder': './new_bgr_reg_datasets/train/images23/',
    #  'label_folder': './new_bgr_reg_datasets/train/labels23/'},
      {'image_folder': './new_bgr_reg_datasets/train/images24/',
     'label_folder': './new_bgr_reg_datasets/train/labels24/'},
      {'image_folder': './new_bgr_reg_datasets/train/images25/',
     'label_folder': './new_bgr_reg_datasets/train/labels25/'},
    #   {'image_folder': './new_bgr_reg_datasets/train/images26/',
    #  'label_folder': './new_bgr_reg_datasets/train/labels26/'},
     {'image_folder': './new_bgr_reg_datasets/train/images27/',
     'label_folder': './new_bgr_reg_datasets/train/labels27/'},
     {'image_folder': './new_bgr_reg_datasets/train/images28/',
     'label_folder': './new_bgr_reg_datasets/train/labels28/'},
     {'image_folder': './new_bgr_reg_datasets/train/images29/',
     'label_folder': './new_bgr_reg_datasets/train/labels29/'},
      {'image_folder': './new_bgr_reg_datasets/train/images30/',
     'label_folder': './new_bgr_reg_datasets/train/labels30/'},
     {'image_folder': './new_bgr_reg_datasets/train/images31/',
     'label_folder': './new_bgr_reg_datasets/train/labels31/'},
     {'image_folder': './new_bgr_reg_datasets/train/images32/',
     'label_folder': './new_bgr_reg_datasets/train/labels32/'},
      {'image_folder': './new_bgr_reg_datasets/train/images33/',
     'label_folder': './new_bgr_reg_datasets/train/labels33/'},
      {'image_folder': './new_bgr_reg_datasets/train/images34/',
     'label_folder': './new_bgr_reg_datasets/train/labels34/'},
      {'image_folder': './new_bgr_reg_datasets/train/images35/',
     'label_folder': './new_bgr_reg_datasets/train/labels35/'},
      {'image_folder': './new_bgr_reg_datasets/train/images36/',
     'label_folder': './new_bgr_reg_datasets/train/labels36/'},
     {'image_folder': './new_bgr_reg_datasets/train/images37/',
     'label_folder': './new_bgr_reg_datasets/train/labels37/'},
     {'image_folder': './new_bgr_reg_datasets/train/images38/',
     'label_folder': './new_bgr_reg_datasets/train/labels38/'},
     {'image_folder': './new_bgr_reg_datasets/train/images39/',
     'label_folder': './new_bgr_reg_datasets/train/labels39/'},
      {'image_folder': './new_bgr_reg_datasets/train/images40/',
     'label_folder': './new_bgr_reg_datasets/train/labels40/'},
     {'image_folder': './new_bgr_reg_datasets/train/images41/',
     'label_folder': './new_bgr_reg_datasets/train/labels41/'},
     {'image_folder': './new_bgr_reg_datasets/train/images42/',
     'label_folder': './new_bgr_reg_datasets/train/labels42/'},
      {'image_folder': './new_bgr_reg_datasets/train/images43/',
     'label_folder': './new_bgr_reg_datasets/train/labels43/'},
      {'image_folder': './new_bgr_reg_datasets/train/images44/',
     'label_folder': './new_bgr_reg_datasets/train/labels44/'},
        {'image_folder': './new_bgr_reg_datasets/train/images45/',
     'label_folder': './new_bgr_reg_datasets/train/labels45/'},
        {'image_folder': './new_bgr_reg_datasets/train/images46/',
     'label_folder': './new_bgr_reg_datasets/train/labels46/'},
       {'image_folder': './new_bgr_reg_datasets/train/images47/',
     'label_folder': './new_bgr_reg_datasets/train/labels47/'},
     {'image_folder': './new_bgr_reg_datasets/train/images48/',
     'label_folder': './new_bgr_reg_datasets/train/labels48/'},
     {'image_folder': './new_bgr_reg_datasets/train/images49/',
     'label_folder': './new_bgr_reg_datasets/train/labels49/'},
      {'image_folder': './new_bgr_reg_datasets/train/images50/',
     'label_folder': './new_bgr_reg_datasets/train/labels50/'},
     {'image_folder': './new_bgr_reg_datasets/train/images51/',
     'label_folder': './new_bgr_reg_datasets/train/labels51/'},
     {'image_folder': './new_bgr_reg_datasets/train/images52/',
     'label_folder': './new_bgr_reg_datasets/train/labels52/'},
      {'image_folder': './new_bgr_reg_datasets/train/images53/',
     'label_folder': './new_bgr_reg_datasets/train/labels53/'},
]

val_paths = [
      {'image_folder': './new_bgr_reg_datasets/val/images20/',
     'label_folder': './new_bgr_reg_datasets/val/labels20/'},
     {'image_folder': './new_bgr_reg_datasets/val/images21/',
     'label_folder': './new_bgr_reg_datasets/val/labels21/'},
     {'image_folder': './new_bgr_reg_datasets/val/images22/',
     'label_folder': './new_bgr_reg_datasets/val/labels22/'},
    #   {'image_folder': './new_bgr_reg_datasets/val/images23/',
    #  'label_folder': './new_bgr_reg_datasets/val/labels23/'},
      {'image_folder': './new_bgr_reg_datasets/val/images24/',
     'label_folder': './new_bgr_reg_datasets/val/labels24/'},
      {'image_folder': './new_bgr_reg_datasets/val/images25/',
     'label_folder': './new_bgr_reg_datasets/val/labels25/'},
    #   {'image_folder': './new_bgr_reg_datasets/val/images26/',
    #  'label_folder': './new_bgr_reg_datasets/val/labels26/'},
     {'image_folder': './new_bgr_reg_datasets/val/images27/',
     'label_folder': './new_bgr_reg_datasets/val/labels27/'},
     {'image_folder': './new_bgr_reg_datasets/val/images28/',
     'label_folder': './new_bgr_reg_datasets/val/labels28/'},
     {'image_folder': './new_bgr_reg_datasets/val/images29/',
     'label_folder': './new_bgr_reg_datasets/val/labels29/'},
      {'image_folder': './new_bgr_reg_datasets/val/images30/',
     'label_folder': './new_bgr_reg_datasets/val/labels30/'},
     {'image_folder': './new_bgr_reg_datasets/val/images31/',
     'label_folder': './new_bgr_reg_datasets/val/labels31/'},
     {'image_folder': './new_bgr_reg_datasets/val/images32/',
     'label_folder': './new_bgr_reg_datasets/val/labels32/'},
      {'image_folder': './new_bgr_reg_datasets/val/images33/',
     'label_folder': './new_bgr_reg_datasets/val/labels33/'},
      {'image_folder': './new_bgr_reg_datasets/val/images34/',
     'label_folder': './new_bgr_reg_datasets/val/labels34/'},
      {'image_folder': './new_bgr_reg_datasets/val/images35/',
     'label_folder': './new_bgr_reg_datasets/val/labels35/'},
      {'image_folder': './new_bgr_reg_datasets/val/images36/',
     'label_folder': './new_bgr_reg_datasets/val/labels36/'},
     {'image_folder': './new_bgr_reg_datasets/val/images37/',
     'label_folder': './new_bgr_reg_datasets/val/labels37/'},
     {'image_folder': './new_bgr_reg_datasets/val/images38/',
     'label_folder': './new_bgr_reg_datasets/val/labels38/'},
     {'image_folder': './new_bgr_reg_datasets/val/images39/',
     'label_folder': './new_bgr_reg_datasets/val/labels39/'},
      {'image_folder': './new_bgr_reg_datasets/val/images40/',
     'label_folder': './new_bgr_reg_datasets/val/labels40/'},
     {'image_folder': './new_bgr_reg_datasets/val/images41/',
     'label_folder': './new_bgr_reg_datasets/val/labels41/'},
     {'image_folder': './new_bgr_reg_datasets/val/images42/',
     'label_folder': './new_bgr_reg_datasets/val/labels42/'},
      {'image_folder': './new_bgr_reg_datasets/val/images43/',
     'label_folder': './new_bgr_reg_datasets/val/labels43/'},
      {'image_folder': './new_bgr_reg_datasets/val/images44/',
     'label_folder': './new_bgr_reg_datasets/val/labels44/'},
      {'image_folder': './new_bgr_reg_datasets/val/images45/',
     'label_folder': './new_bgr_reg_datasets/val/labels45/'},
        {'image_folder': './new_bgr_reg_datasets/val/images46/',
     'label_folder': './new_bgr_reg_datasets/val/labels46/'},
     {'image_folder': './new_bgr_reg_datasets/train/images47/',
     'label_folder': './new_bgr_reg_datasets/train/labels47/'},
     {'image_folder': './new_bgr_reg_datasets/train/images48/',
     'label_folder': './new_bgr_reg_datasets/train/labels48/'},
     {'image_folder': './new_bgr_reg_datasets/train/images49/',
     'label_folder': './new_bgr_reg_datasets/train/labels49/'},
      {'image_folder': './new_bgr_reg_datasets/train/images50/',
     'label_folder': './new_bgr_reg_datasets/train/labels50/'},
     {'image_folder': './new_bgr_reg_datasets/train/images51/',
     'label_folder': './new_bgr_reg_datasets/train/labels51/'},
     {'image_folder': './new_bgr_reg_datasets/train/images52/',
     'label_folder': './new_bgr_reg_datasets/train/labels52/'},
      {'image_folder': './new_bgr_reg_datasets/train/images53/',
     'label_folder': './new_bgr_reg_datasets/train/labels53/'},
]
# train_paths = [
#       {'image_folder': './etc_ocr_reg_datasets/train/images1/',
#      'label_folder': './etc_ocr_reg_datasets/train/labels1/'},

# ]

# val_paths = [
#       {'image_folder': './etc_ocr_reg_datasets/val/images1/',
#      'label_folder': './etc_ocr_reg_datasets/val/labels1/'},

# ]

paths = train_paths
# paths = val_paths
# Đường dẫn đến file LMDB sẽ được tạo
# lmdb_path = "./ocr_reg_lmdb/train/"
lmdb_path = "./ocr_reg_148_lmdb/train/"
# Mở file LMDB để ghi dữ liệu
env = lmdb.open(lmdb_path, map_size=int(1e12))


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


# Khởi tạo transaction để ghi dữ liệu
with env.begin(write=True) as txn:
    # Duyệt qua các file ảnh trong folder
    cnt = 1
    cache = {}
    for path in paths:
        image_folder = path['image_folder']
        label_folder = path['label_folder']
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

            # Ghi dữ liệu vào LMDB
            imageKey = 'image-%09d'.encode() % cnt
            labelKey = 'label-%09d'.encode() % cnt
            txn.put(imageKey, imageBin)
            txn.put(labelKey, label.encode())
            cnt += 1

    txn.put('num-samples'.encode(), str(cnt-1).encode())

# Đóng file LMDB
env.close()