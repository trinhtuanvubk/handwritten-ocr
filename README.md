# Vietnamese Handwritten OCR

### Environment

- To create virtual environment:
```
vitualenv venv
pip install -r requiments.txt
```

### Data
- To create lmdb data from raw data:
```bash
python3 main.py --scenario create_lmdb_data \
--raw_data_path "./data/OCR/training_data" \
--raw_data_type "folder" \
--data_mode "train" \
--lmdb_data_path "./data/kalapa_lmdb/"
```

- Flag:
    - `raw_data_path`: path to raw data
    - `raw_data_type`: have 3 values:
        - `json`: a dir contains image and a json file with each line contains path to image and text label.
        - `folder`: a dir contains image subdirs and a dir contains subfile .txt label.
        - `other`: the second gen type from my repo [OCR-Vietnamese-Text-Generator](https://github.com/trinhtuanvubk/OCR-Vietnamese-Text-Generator)
    - `data_mode`: train data or eval data
    - `lmdb_data_path`: path to output lmdb data

### Train
- To run training:
```bash
python3 main.py --scenario train \
--model SVTR \
--lmdb_data_path "./data/kalapa_lmdb/"
--batch_size 16
--num_epoch 1000
```

- Flag:
    - `model`: select model (`SVTR`, `PPLCNETV3`). just test on SVTR now

### Inference
- To run inference test:
```bash
python3 main.py --scenario infer --image_test_path "/home/sangdt/research/voice/svtr-pytorch/data/OCR/public_test/images/14/0.jpg"
```

### Submission

```bash
python3 submission.py
```

### Todo
- `scenario`: - prepare_data: convert image to GRAY by opencv

- `nnet`: find the way to remove hard code: input shape and output max length
- Merge config from `dataloader/config.yaml` and `utils/args.py`

### Note
- See `dataloader/config.yaml` to config augmentation, normalization and preprocessing. 
- See `utils/args.py` to modify some config
- Some hard code at set max text length to the last layer in  `nnet/modules/rec_head`
