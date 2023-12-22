# Vietnamese Handwritten OCR (Top 5 Kalapa Challenge 2023)
- In this project, we implement a lightweight model `SVTR` in pytorch for handwritten-ocr task. In terms of steps, we trained a pretrained model with a large of generated data. Then, we fintuned model with real handwritten data.
### Environment

- To create virtual environment:
```
vitualenv venv
pip install -r requiments.txt
```

### Data
- To generate data, you can use some handwritten fonts and the text corpus to generate with my repo [OCR-Vietnamese-Text-Generator](https://github.com/trinhtuanvubk/OCR-Vietnamese-Text-Generator) (or the original repo). I also provided some address corpus files in `data/corpus` for your reference.
- Note: data format should be in 2 type (a folder contains all images and a folder contains all label text files)
```
|___data
|    |___train
|    |    |___images
|    |    |    |___0.jpg
|    |    |    |___...
|    |    |___labels
|    |    |    |___0.txt
|    |    |    |___...
|    |___val
|    |    |___images
|    |    |    |___0.jpg
|    |    |    |___...
|    |    |___labels
|    |    |    |___0.txt
|    |    |    |___...
```

- To preprocess (crop image):
```bash
python3 main.py --scenario preprocess \
--raw_data_path "./data/OCR/training_data"
```

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
        - `other`: the second gen type from my repo.
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
python3 main.py --scenario infer --image_test_path "path/to/image.jpg"
```

### Export Onxx
- To export model to onnx (optional):
```
python3 export_onnx.py
```

### Submission
- To run infer with a folder:
    - run in batch:
        ```bash
        python3 submission.py
        ```
    - run each image:
        ```bash
        python3 torch_submission.py
        ```
    - run each image with onnx:
        ```bash
        python3 onnx_submission.py
        ```

### Todo
- [ ] Implement SAR loss that helps training model with multiple loss
- [ ] Implement LCNET
- [ ] `nnet`: find the way to remove hard code: input shape and output max length
- [ ] Merge config from `dataloader/config.yaml` and `utils/args.py`

### Note
- See `dataloader/config.yaml` to config augmentation, normalization and preprocessing. 
- See `utils/args.py` to modify some config
- Hard code at set max text length to the last layer in  `nnet/modules/rec_head`
- Hard code at `T_max` in cosine lr schedualer
- To build ngram model: https://github.com/kmario23/KenLM-training
- I had a mistake when building dictionary that duplicates 2 symbols. I dont have the resource to retrain model, so comment warning in `python3.10/site-packages/pyctcdecode/alphabet.py` to pass the duplicate check. Please use `utils/vi_dict_fix.txt` to avoid this mistake.
