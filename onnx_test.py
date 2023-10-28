import torch
import onnx
import onnxruntime
import cv2
from utils.util import *

path = "test_onnx.onnx"
img_path = "./data/public_test/images/31/1.jpg"
sess = onnxruntime.InferenceSession(path, providers=['CUDAExecutionProvider'])

print([x.name for x in sess.get_inputs()])
print([x.shape for x in sess.get_inputs()])

# output infro
print([x.name for x in sess.get_outputs()])
print([x.shape for x in sess.get_outputs()])


imgC, imgH, imgW = (3,48,720)
max_wh_ratio = imgW / imgH

image = cv2.imread(img_path)
norm_img = resize_norm_img(image, max_wh_ratio)
print(norm_img.shape)

