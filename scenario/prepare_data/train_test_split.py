import os
from pathlib import Path
import random
import shutil

def train_test_split(args):
    # image_folder = os.path.join(args.raw_data_path, 'images')\
    if args.raw_data_type==0:
        label_folder = os.path.join(args.raw_data_path, 'annotations')

        # eval_images = os.path.join("data/OCR", "eval/images")
        # os.makedirs(eval_images, exist_ok=True)
        eval_annotations = os.path.join(args.raw_data_path, "eval_annotations")
        os.makedirs(eval_annotations, exist_ok=True)

        train_annotations = os.path.join(args.raw_data_path, "train_annotations")
        os.makedirs(train_annotations, exist_ok=True)

        with open(os.path.join(eval_annotations, "eval_text.txt"), 'w') as eval_label:
            with open(os.path.join(train_annotations, "train_text.txt"), 'w') as train_label:
                for labels in os.listdir(label_folder):
                    ran_list = random.sample(range(0, 27), 3)
                    print(ran_list)
                    with open(os.path.join(label_folder, labels), "r") as l:
                        print(os.path.join(label_folder, labels))
                        l_data = l.readlines()
                        print(l_data)
                        for i, line in enumerate(l_data):
                            print(i)
                            if i in ran_list:
                                eval_label.write(l_data[i])
                            else:
                                train_label.write(line)

    else:
        
        val_images = os.path.join(args.raw_data_path, "val", "images")
        val_labels = os.path.join(args.raw_data_path, "val", "labels")

        os.makedirs(val_images, exist_ok=True)
        os.makedirs(val_labels, exist_ok=True)

        train_images = os.path.join(args.raw_data_path, "train", "images")
        train_labels = os.path.join(args.raw_data_path, "train", "labels")

        os.makedirs(train_images, exist_ok=True)
        os.makedirs(train_labels, exist_ok=True)


        labels = os.path.join(args.raw_data_path, "labels")
        images = os.path.join(args.raw_data_path, "images")

        len_ = len(os.listdir(images))

        ran_val = random.sample(range(0, len_), int(0.15*len_))

        for i, lal in enumerate(os.listdir(labels)):
            lal_path = os.path.join(labels, lal)
            lal_name = lal.split(".")[0]
            image_path = os.path.join(images, f"{lal_name}.jpg")
            if i in ran_val:
                shutil.copy(image_path, val_images)
                shutil.copy(lal_path, val_labels)
            else:
                shutil.copy(image_path, train_images)
                shutil.copy(lal_path, train_labels)

            

