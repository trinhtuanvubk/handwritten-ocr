import csv
import shutil
import os


# public_label = "./711_0.3lmSon.csv"
public_label = "./best_check.csv"
my_pred = "./811_Son_both.csv"
pair_dict = {}

with open(public_label, "r") as pl:
    csv_reader = csv.reader(pl, delimiter=',')
    for row in csv_reader:
        try:
            text = row[2]
            image = row[1]
            pair_dict[image] = [text]
        except:
            text = row[1]
            image = row[0]
            pair_dict[image] = [text]

            
with open(my_pred, "r") as mp:

    csv_reader = csv.reader(mp, delimiter=',')
    for row in csv_reader:
        text = row[1]
        image = row[0]
        try:
            pair_dict[image].append(text)
        except:
            continue

# print(pair_dict)

count = 0
for image, pair in pair_dict.items():
    if pair[0].strip() != pair[1].strip():
        print(f"pred : {pair[1]}")
        print(f"label: {pair[0]}")
        print(f"---------{count} | {image}-----------")
        count+=1


# 31/23.jpg