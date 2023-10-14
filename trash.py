import onnxruntime
import onnx
import numpy as np
from utils import util, config

import cv2
import sys
import os

# sys.path.append('../')
# from paddleocr_dev import PaddleOCR, draw_ocr
# from ppocr.data import create_operators, transform
# from ppocr.postprocess import build_post_process
from process_data.data import create_operators, transform
from process_data.postprocess import build_post_process
from utils.draw import draw_ocr

onnx_card_det = "./onnx_repository/card_det.onnx"
onnx_text_det = "./onnx_repository/text_det.onnx"
onnx_text_rec = "./onnx_repository/text_rec.onnx"


image_path = "./images/vutt_input/f1.jpeg"


def load_onnx(onnx_path, providers='CPUExecutionProvider'):
    sess = onnxruntime.InferenceSession(onnx_path, providers=[providers])
    return sess

card_det = load_onnx(onnx_card_det)
text_det = load_onnx(onnx_text_det)
text_rec = load_onnx(onnx_text_rec)

# inputs = [x.name for x in text_rec.get_inputs()]
# outputs = [x.name for x in text_rec.get_outputs()]
# print(inputs)
# print(outputs)
# print(text_rec.get_inputs()[0].shape)


image = cv2.imread(image_path)
image = util.check_img(image)
# image = [image]
print(image.shape)

card_det_pre_process_list = config["card_det"]["pre_process_list"]

card_det_preprocess_op = create_operators(card_det_pre_process_list)

card_det_postprocess_params = config["card_det"]["postprocess_params"]
card_det_postprocess_op = build_post_process(card_det_postprocess_params)

# ori_im = image.copy()
data = {'image': image}
data = transform(data, card_det_preprocess_op)
img, shape_list = data
print(img.shape, shape_list)
img = np.expand_dims(img, axis=0)
shape_list = np.expand_dims(shape_list, axis=0)
img = img.copy()

# model = onnx.load(onnx_path)
# inputs = [x for x in sess.get_inputs()]
outputs = [x.name for x in card_det.get_outputs()]
# print(inputs)
# print(outputs)
# print(sess.get_inputs()[0].shape)

input_dict = {"x":img}
print(img.shape)
outputs = card_det.run(outputs, input_dict)

preds = {"maps":outputs[0]}
post_result = card_det_postprocess_op(preds, shape_list)
dt_boxes = post_result[0]['points']
print(dt_boxes)
img_show = draw_ocr(image, dt_boxes, None, None, font_path="/home/ai22/Documents/VUTT/PaddleOCR/doc/fonts/SVN-Arial3.ttf")
cv2.imwrite("./images/vutt_output/step1.jpg", img_show)

if len(dt_boxes)==1:
    box = dt_boxes[0]
    
card_image = util.get_rotate_crop_image(image, np.array(box, np.float32))
card_image = util.check_img(card_image)
print(card_image.shape)

print("DONE 1")
# -----------------------------------------------------------------------------------
print(card_image.shape)
text_det_pre_process_list = config["text_det"]["pre_process_list"]

text_det_preprocess_op = create_operators(text_det_pre_process_list)

text_det_postprocess_params = config["text_det"]["postprocess_params"]
text_det_postprocess_op = build_post_process(text_det_postprocess_params)

card_data = {'image': card_image}
card_data = transform(card_data, text_det_preprocess_op)
card_im, card_shape_list = card_data
print(card_im.shape, card_shape_list)
card_im = np.expand_dims(card_im, axis=0)
card_shape_list = np.expand_dims(card_shape_list, axis=0)


card_input_dict = {"x":card_im}
print(card_im.shape)
textdet_outputs = [x.name for x in text_det.get_outputs()]
print(text_det.get_inputs()[0].shape)
textdet_outputs = text_det.run(textdet_outputs, card_input_dict)
print(textdet_outputs[0].shape)

textdetpreds = {"maps":textdet_outputs[0]}
text_post_result = text_det_postprocess_op(textdetpreds, card_shape_list)
textdet_dt_boxes = text_post_result[0]['points']
print(textdet_dt_boxes)
textdet_img_show = draw_ocr(card_image, textdet_dt_boxes, None, None, font_path="/home/ai22/Documents/VUTT/PaddleOCR/doc/fonts/SVN-Arial3.ttf")
cv2.imwrite("./images/vutt_output/step2.jpg", textdet_img_show)

print("DONE 2")
# --------------------------------------------------------------------------
img_crop_list = []
textdet_dt_boxes = util.sorted_boxes(textdet_dt_boxes)

for i in range(len(textdet_dt_boxes)):
    text_img_crop = util.get_rotate_crop_image(card_image, np.array(textdet_dt_boxes[i], np.float32))
    img_crop_list.append(text_img_crop)

print(len(img_crop_list))

text_rec_postprocess_params = config["text_rec"]["postprocess_params"]
text_rec_postprocess_op = build_post_process(text_rec_postprocess_params)
img_list = img_crop_list
img_num = len(img_list)
# Calculate the aspect ratio of all text bars
width_list = []
for img in img_list:
    width_list.append(img.shape[1] / float(img.shape[0]))

# Sorting can speed up the recognition process
indices = np.argsort(np.array(width_list))
rec_res = [['', 0.0]] * img_num
batch_num = 6

for beg_img_no in range(0, img_num, batch_num):
    end_img_no = min(img_num, beg_img_no + batch_num)
    norm_img_batch = []
    imgC, imgH, imgW = (3,48,320)
    max_wh_ratio = imgW / imgH
    for ino in range(beg_img_no, end_img_no):
        h, w = img_list[indices[ino]].shape[0:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
    for ino in range(beg_img_no, end_img_no):
        norm_img = util.resize_norm_img(img_list[indices[ino]],
                                                    max_wh_ratio)
        norm_img = norm_img[np.newaxis, :]
        norm_img_batch.append(norm_img)

    norm_img_batch = np.concatenate(norm_img_batch)
    norm_img_batch = norm_img_batch.copy()
    print(norm_img_batch.shape)
    rec_input_dict = {'x': norm_img_batch}
    text_outputs = [x.name for x in text_rec.get_outputs()]
    print(text_rec.get_inputs()[0].shape)
    textrec_outputs = text_rec.run(text_outputs, rec_input_dict)

    text_preds = textrec_outputs[0]

    print(text_preds.shape)

    rec_result = text_rec_postprocess_op(text_preds)
    print(rec_result)

    for rno in range(len(rec_result)):
        rec_res[indices[beg_img_no + rno]] = rec_result[rno]
print(rec_res, len(rec_res))

filter_boxes, filter_rec_res = [], []
for box, rec_result in zip(textdet_dt_boxes, rec_res):
    text, score = rec_result
    if score >= 0.5:
        filter_boxes.append(box)
        filter_rec_res.append(rec_result)

result = [[box.tolist(), res] for box, res in zip(filter_boxes, filter_rec_res)]
print(result)
boxes = [line[0] for line in result]
txts = [line[1][0].strip() for line in result]
scores = [line[1][1] for line in result]
img_show = draw_ocr(card_image, boxes, txts, scores, font_path="/home/ai22/Documents/VUTT/PaddleOCR/doc/fonts/SVN-Arial3.ttf")
cv2.imwrite("images/vutt_output/last_result.jpg", img_show)
print("DONE 3")





