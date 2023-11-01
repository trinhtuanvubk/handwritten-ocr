import os
import shutil
import random


folder_in = "/home/sonlt373/Desktop/SoNg/OCR_handwriting_shop_sticker/data/data_sticker_handwriting/KALAPA_ByteBattles_2023_OCR_Set1/OCR/kalapa_training_set/kalapa_train_fixed_bak/images"


list_image_in = [file for file in os.listdir(folder_in) if file.endswith('.jpg') or file.endswith('.png')]
# list_image_in = list(set(list_image_in) - set(os.listdir("/home/sonlt373/Desktop/SoNg/yolov8_barcode/AnhFlashScannerDaiTu_split600/AnhFlashScannerDaiTu_split500")))
list_image_out  = random.sample(list_image_in, k=int(300))

for path in list_image_out:
    shutil.copy(os.path.join(folder_in, path), "/home/sonlt373/Desktop/SoNg/OCR_handwriting_shop_sticker/data/data_sticker_handwriting/KALAPA_ByteBattles_2023_OCR_Set1/OCR/kalapa_training_set/augment_011123/images")
    shutil.copy(os.path.join("/home/sonlt373/Desktop/SoNg/OCR_handwriting_shop_sticker/data/data_sticker_handwriting/KALAPA_ByteBattles_2023_OCR_Set1/OCR/kalapa_training_set/kalapa_train_fixed_bak/labels", path[:-4] + '.txt'), "/home/sonlt373/Desktop/SoNg/OCR_handwriting_shop_sticker/data/data_sticker_handwriting/KALAPA_ByteBattles_2023_OCR_Set1/OCR/kalapa_training_set/augment_011123labels")

