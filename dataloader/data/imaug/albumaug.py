import cv2
import albumentations as A
import numpy as np
# import math
# import os
# import random
# from tqdm import tqdm
# import concurrent.futures
# import argparse

class Albumentation(object):
    def __init__(self, **kwargs):
        self.transform = A.Compose([
        A.Blur(p = 0.1),
        A.MotionBlur(p = 0.1),
        A.GaussNoise(p = 0.1),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p = 0.2),
        A.CLAHE(p = 0.2),
        A.RandomBrightnessContrast(p=0.7),
        A.ISONoise(p = 0.3),
        A.RGBShift(p = 0.5),
        A.ChannelShuffle(p = 0.5),
        A.ShiftScaleRotate( shift_limit = 0.0, shift_limit_y = 0.01, scale_limit=0.0, rotate_limit=1, p=0.2),
        # A.ElasticTransform(alpha_affine=0.5, alpha=0.5, sigma=0, p = 0.1),
        # A.Perspective(p = 0.2),
        A.ToGray(p = 0.25),
        ])
    
    def __call__(self, data):
        img = data['image']
        transformed = self.transform(image=img)
        transformed_image = transformed['image']

        data['image'] = transformed_image
        return data




# def save_dict2txt(results_dict, out_path):
#     with open(out_path, "a", encoding="utf-8") as outfile:  # Use "a" mode for appending
#         for key, value in results_dict.items():
#             line = f"{key}\t{value}"
#             outfile.write(line)
#     print("=======================================append dict done=====================================================")
#     return "append dict done"

# def pad_image(image, padding_width_percent, padding_height_percent, padding_color=(0, 0, 0)):
#     """
#     Pad an image with a specified percentage of padding width and height.

#     Parameters:
#     - image: The input image as a NumPy array.
#     - padding_width_percent: The percentage of padding width (0 to 100).
#     - padding_height_percent: The percentage of padding height (0 to 100).
#     - padding_color: The color of the padding in BGR format (default is black).

#     Returns:
#     - padded_image: The image with padding and the original image randomly positioned.
#     """
#     if padding_width_percent < 0 or padding_height_percent < 0:
#         raise ValueError("Padding percentages should be non-negative.")

#     height, width, channels = image.shape

#     # Calculate the amount of padding in pixels based on the percentages
#     pad_width = int(padding_width_percent / 100 * width)
#     pad_height = int(padding_height_percent / 100 * height)

#     # Create an image with the desired padding color
#     padded_image = np.full((height + 2 * pad_height, width + 2 * pad_width, channels), padding_color, dtype=np.uint8)

#     # Generate random positions for the top-left corner of the original image within the padded canvas
#     rand_x = random.randint(0, 2 * pad_width)
#     rand_y = random.randint(0, 2 * pad_height)

#     # Calculate the coordinates for pasting the original image
#     paste_x1 = max(0, rand_x)
#     paste_x2 = min(rand_x + width, padded_image.shape[1])
#     paste_y1 = max(0, rand_y)
#     paste_y2 = min(rand_y + height, padded_image.shape[0])

#     # Calculate the coordinates for copying a portion of the original image
#     copy_x1 = max(0, -rand_x)
#     copy_x2 = min(width, padded_image.shape[1] - rand_x)
#     copy_y1 = max(0, -rand_y)
#     copy_y2 = min(height, padded_image.shape[0] - rand_y)

#     # Paste the original image into the padded canvas at the random position
#     padded_image[paste_y1:paste_y2, paste_x1:paste_x2] = image[copy_y1:copy_y2, copy_x1:copy_x2]

#     return padded_image


# def augmentation(image):
#     # pixel_color = image[3, 3]
#     # try:
#     #     if (int(pixel_color[0]) + int(pixel_color[1]) + int(pixel_color[2]))/3 < 100:
#     #         pixel_color = (128,128,128)
#     # except:
#     #     pixel_color = (128,128,128)

#     # image = pad_image(image, 5, 10, padding_color=pixel_color)

#         # cv2.imwrite("check.jpg", image)
#     # image = cv2.resize(image, (int(image.shape[1] * 64 / image.shape[0]), 64 ))    # 48

#     transform = A.Compose([
#     A.Blur(p = 0.1),
#     A.MotionBlur(p = 0.1),
#     A.GaussNoise(p = 0.1),
#     A.GridDistortion( num_steps=5, distort_limit=0.1, p = 0.2),
#     A.CLAHE(p = 0.2),
#     A.RandomBrightnessContrast(p=0.7),
#     A.ISONoise(p = 0.3),
#     A.RGBShift(p = 0.5),
#     A.ChannelShuffle(p = 0.5),
#     A.ShiftScaleRotate( shift_limit = 0.0, shift_limit_y = 0.01, scale_limit=0.0, rotate_limit=1, p=0.2),
#     # A.ElasticTransform(alpha_affine=0.5, alpha=0.5, sigma=0, p = 0.1),
#     # A.Perspective(p = 0.2),
#     A.ToGray(p = 0.25),
#     ])
        
#     image_aug_list = []
#     for _ in range(5):

#         # Randomcrop right of image
#         # random_crop =  random.uniform(0,1)
#         # if random_crop > 0.6:
#         # # Define the desired height and width for the cropped image
#         #     desired_height = image.shape[0]  # Keep the original height
#         #     desired_width = int(image.shape[1] * random.uniform(0.95,1))  # Crop random % from the right


# #     # # Randomly choose the top-left corner for cropping
#         #     # y_start = np.random.randint(0, image.shape[0] - desired_height + 1)
#         #     # x_start = np.random.randint(0, image.shape[1] - desired_width + 1)

#         #     # Crop the image
#         #     image = image[0 : desired_height, 0 : 0 + desired_width]

#         transformed = transform(image=image)
#         transformed_image = transformed['image']

#         image_aug_list.append(transformed_image)
#     return image_aug_list, transformed_image


# def process_line(img_path_input):
#     img = cv2.imread(img_path_input)

#     image_aug_list = augmentation(img)

#     return image_aug_list

# # Your input and output paths and other variables
# root = "/home/sonlt373/Desktop/SoNg/OCR_handwriting_shop_sticker/data/data_sticker_handwriting/KALAPA_ByteBattles_2023_OCR_Set1/OCR/kalapa_training_set/kalapa_train_fixed/"
# input_path = os.path.join(root, "images")
# anno_input_path = os.path.join(root, "labels")


# # Function to process a single image
# def process_image(root, image_path, name, annotation):
#     input_path = os.path.join(root, "images")
#     anno_input_path = os.path.join(root, "labels")
#     image_aug_list, transformed_image = process_line(image_path)

#     # if want save list images augmented
#     # for i, image_aug in enumerate(image_aug_list):
#     #     image_path_out = os.path.join(input_path, name + f'_aug_{i}.jpg')
#     #     anno_path_out = os.path.join(anno_input_path, name + f'_aug_{i}.txt')
#     #     cv2.imwrite(image_path_out, image_aug)
#     #     with open(anno_path_out, 'w') as f:
#     #         f.write(annotation)

#     #   If want save one image augmented
#     image_path_out = os.path.join(input_path, name + f'.jpg')
#     anno_path_out = os.path.join(anno_input_path, name + f'.txt')
#     cv2.imwrite(image_path_out, transformed_image)
#     with open(anno_path_out, 'w') as f:
#         f.write(annotation)

# def main(root):
#     input_path = os.path.join(root, "images")
#     anno_input_path = os.path.join(root, "labels")

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         for path_anno in tqdm(os.listdir(anno_input_path)):
#             name = path_anno[:-4]
#             anno_path = os.path.join(anno_input_path, path_anno)
#             image_path = os.path.join(input_path, name + '.jpg')

#             with open(anno_path, 'r') as f:
#                 annotation = f.read()

#             executor.submit(process_image, root, image_path, name, annotation)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Image processing script")
#     parser.add_argument("--root", required=True, default= '/home/sonlt373/Desktop/SoNg/OCR_handwriting_shop_sticker/data/data_sticker_handwriting/KALAPA_ByteBattles_2023_OCR_Set1/OCR/kalapa_training_set/kalapa_train_fixed/', help="Root path for input and output")
#     args = parser.parse_args()
#     main(args.root)
#     print("-----------------------------------------------------------")
