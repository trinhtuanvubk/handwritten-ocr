import os
from pathlib import Path
import random
import shutil

def train_test_split(args):
    # image_folder = os.path.join(args.raw_data_path, 'images')
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
