import numpy as np
import cv2
import requests
import tqdm
from loguru import logger
import sys
import os

from PIL import Image
import math
from tqdm import tqdm

def check_and_read(img_path):
    if os.path.basename(img_path)[-3:] in ['gif', 'GIF']:
        gif = cv2.VideoCapture(img_path)
        ret, frame = gif.read()
        if not ret:
            
            return None, False
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        imgvalue = frame[:, :, ::-1]
        return imgvalue, True, False
    return None, False, False

def is_link(s):
    return s is not None and s.startswith('http')

def download_with_progressbar(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size_in_bytes = int(response.headers.get('content-length', 1))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    else:
        logger.error("Something went wrong while downloading models")
        sys.exit(0)



def img_decode(content: bytes):
    np_arr = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def check_img(img):
    if isinstance(img, bytes):
        img = img_decode(img)
    if isinstance(img, str):
        # download net image
        if is_link(img):
            download_with_progressbar(img, './images/api_images/tmp.jpg')
            img = './images/api_images/tmp.jpg'
        image_file = img
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            with open(image_file, 'rb') as f:
                img = img_decode(f.read())
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            return None
    if isinstance(img, np.ndarray) and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops

def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data



def get_rotate_crop_image(img, points):
    # Use Green's theory to judge clockwise or counterclockwise
    # author: biyanhua
    d = 0.0
    for index in range(-1, 3):
        d += -0.5 * (points[index + 1][1] + points[index][1]) * (
                points[index + 1][0] - points[index][0])
    if d < 0:  # counterclockwise
        tmp = np.array(points)
        points[1], points[3] = tmp[3], tmp[1]

    try:
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img
    except Exception as e:
        print(e)



def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def resize_norm_img(img, max_wh_ratio):

    imgC, imgH, imgW = (3,48,720)
    assert imgC == img.shape[2]
    imgW = int((imgH * max_wh_ratio))
    # imgW = imgH * img.shape[1] / img.shape[0]
    # if self.use_onnx:
    #     w = self.input_tensor.shape[3:][0]
    #     if isinstance(w, str):
    #         pass
    #     elif w is not None and w > 0:
    #         imgW = w
    h, w = img.shape[:2]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        # print("real shape is larger")
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
        # print("model shape is larger")
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    # padding_im = -1.0 * np.ones((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    # print(padding_im.shape)
    return padding_im

def resize_norm_img_v2(img, max_wh_ratio):

    imgC, imgH, imgW = (3,48,720)
    assert imgC == img.shape[2]
    imgW = int((imgH * max_wh_ratio))
    h, w = img.shape[:2]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        # print("real shape is larger")
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
        # print("model shape is larger")
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    temp_resized_image = resized_image
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    # padding_im = -1.0 * np.ones((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    # print(padding_im.shape)
    return padding_im, resized_w, temp_resized_image

def check_file(path:str, extention_task = None):
    name = path.split(".")[-2]
    new_name = name
    if extention_task is not None:
        new_name = name+str(extention_task)
    if os.path.isfile(path):
        if extention_task is not None:
            new_name = new_name+"_"

    out_path = path.replace(name, new_name)

    return out_path

def resize_norm_img_svtr(self, img, image_shape):

    imgC, imgH, imgW = image_shape
    resized_image = cv2.resize(
        img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
    resized_image = resized_image.astype('float32')
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    return resized_image

# def resize_norm_img(img,
#                     image_shape,
#                     padding=True,
#                     interpolation=cv2.INTER_LINEAR):
#     imgC, imgH, imgW = image_shape
#     h = img.shape[0]
#     w = img.shape[1]
#     if not padding:
#         resized_image = cv2.resize(
#             img, (imgW, imgH), interpolation=interpolation)
#         resized_w = imgW
#     else:
#         ratio = w / float(h)
#         if math.ceil(imgH * ratio) > imgW:
#             resized_w = imgW
#         else:
#             resized_w = int(math.ceil(imgH * ratio))
#         resized_image = cv2.resize(img, (resized_w, imgH))
#     resized_image = resized_image.astype('float32')
#     if image_shape[0] == 1:
#         resized_image = resized_image / 255
#         resized_image = resized_image[np.newaxis, :]
#     else:
#         resized_image = resized_image.transpose((2, 0, 1)) / 255
#     resized_image -= 0.5
#     resized_image /= 0.5
#     padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
#     padding_im[:, :, 0:resized_w] = resized_image
#     valid_ratio = min(1.0, float(resized_w / imgW))
#     return padding_im, valid_ratio
