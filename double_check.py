import csv
import shutil
import os


public_label = "./data/public_test/kalapa_public_test.csv"
my_pred = "./111_ngram_2.csv"
pair_dict = {}

with open(public_label, "r") as pl:
    csv_reader = csv.reader(pl, delimiter=',')
    for row in csv_reader:
        text = row[2]
        image = row[1]
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

print(pair_dict)

count = 0
for image, pair in pair_dict.items():
    if pair[0].strip() != pair[1].strip():
        print(f"pred : {pair[1]}")
        print(f"label: {pair[0]}")
        print(f"---------{count} | {image}-----------")
        count+=1

'''
NOTE: ụy ụy 
Bà rịa - Br
viết thường
chung cư - CC
thị trấn - TT
K/thiên
'''