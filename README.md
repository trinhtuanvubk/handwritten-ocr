# Handwritten OCR

### Environment

- To create virtual environment:
```
vitualenv venv
pip install -r requiments.txt
```

### Data
```
python3 main.py --create_lmdb_data
```

### Train
```
python3 main.py --scenario train --model SVTR
```

### Test

### Submission

```
python3 submission.py
```

### TODO
- `scenario`: - prepare_data: convert image to GRAY by opencv
              - train: 
              - test: write infer code to run public test

-  `nnet`: find the way to remove hard code: input shape and output max length

### Note
- See `utils/args.py` to modify config
- Some hard code at set max text length to the last layer in  `nnet/modules/rec_head`
- There are 2 types of `args.raw_data_type`: json and folder