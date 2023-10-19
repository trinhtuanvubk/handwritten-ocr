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

### Eval

### Note
- See `utils/args.py` to modify config
- Some hard code at set max text length to the last layer in  `nnet/modules/rec_head`
- There are 2 types of `args.raw_data_type`: json and folder