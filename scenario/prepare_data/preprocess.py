import os
import cv2
from utils.preprocess import detect_text_lines
from tqdm import tqdm

def crop_data(args):
    # this code is used for data_mode: 2
    new_data_path = f"{args.raw_data_path}_crop_raw"
    os.makedirs(new_data_path, exist_ok=True)
    if args.raw_data_type not in [0,1]:
        try:
            for fol in os.listdir(os.path.join(args.raw_data_path, "train")):
                
                if fol.startswith("images"):
                    
                    fol_path = os.path.join(args.raw_data_path, "train", fol)
                    new_fol_path = os.path.join(new_data_path, "train", fol)
                    os.makedirs(new_fol_path, exist_ok=True)

                    for i, img in tqdm(enumerate(os.listdir(fol_path))):
                        
                        img_path = os.path.join(fol_path, img)
                        print(img_path)
                        image = cv2.imread(img_path)
                        image = detect_text_lines(image)
                        if image is None:
                            print("None Image")
                            continue
                        new_img_path = os.path.join(new_fol_path, img)
                        print(new_img_path)
                        cv2.imwrite(new_img_path, image)

            for fol in os.listdir(os.path.join(args.raw_data_path, "val")):
                if fol.startswith("images"):
                    
                    fol_path = os.path.join(args.raw_data_path, "val", fol)
                    new_fol_path = os.path.join(new_data_path, "val", fol)
                    os.makedirs(new_fol_path, exist_ok=True)

                    for i, img in tqdm(enumerate(os.listdir(fol_path))):
                        
                        img_path = os.path.join(fol_path, img)
                        print(img_path)
                        image = cv2.imread(img_path)
                        image = detect_text_lines(image)
                        if image is None:
                            print("None Image")
                            continue
                        new_img_path = os.path.join(new_fol_path, img)
                        print(new_img_path)
                        cv2.imwrite(new_img_path, image)

        except:
            for fol in os.listdir(args.raw_data_path):
                if fol.startswith("images"):
                    fol_path = os.path.join(args.raw_data_path, fol)
                    new_fol_path = os.path.join(new_data_path, fol)
                    os.makedirs(new_fol_path, exist_ok=True)

                    for i, img in tqdm(enumerate(os.listdir(fol_path))):
                        
                        img_path = os.path.join(fol_path, img)
                        image_name = img.split(".")[0]
                        print(img_path)
                        image = cv2.imread(img_path)
                        image = detect_text_lines(image)
                        if image is None:
                            print("None Image")
                            continue
                        new_img_path = os.path.join(new_fol_path, f"{image_name}_crop.jpg")
                        print(new_img_path)
                        cv2.imwrite(new_img_path, image)



