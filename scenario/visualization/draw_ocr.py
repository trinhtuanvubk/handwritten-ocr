# import numpy as np
# import cv2

# def resize_img(img, input_size=600):
#     """
#     resize img and limit the longest side of the image to input_size
#     """
#     img = np.array(img)
#     im_shape = img.shape
#     im_size_max = np.max(im_shape[0:2])
#     im_scale = float(input_size) / float(im_size_max)
#     img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
#     return img


# def draw_ocr(image,
#              boxes,
#              txts=None,
#              scores=None,
#              drop_score=0.5,
#              font_path="./scenario/visualization/font/simfang.ttf"):
#     """
#     Visualize the results of OCR detection and recognition
#     args:
#         image(Image|array): RGB image
#         boxes(list): boxes with shape(N, 4, 2)
#         txts(list): the texts
#         scores(list): txxs corresponding scores
#         drop_score(float): only scores greater than drop_threshold will be visualized
#         font_path: the path of font which is used to draw text
#     return(array):
#         the visualized img
#     """
#     if scores is None:
#         scores = [1] * len(boxes)
#     box_num = len(boxes)
#     for i in range(box_num):
#         if scores is not None and (scores[i] < drop_score or
#                                    math.isnan(scores[i])):
#             continue
#         box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
#         image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
#     if txts is not None:
#         img = np.array(resize_img(image, input_size=600))
#         txt_img = text_visual(
#             txts,
#             scores,
#             img_h=img.shape[0],
#             img_w=600,
#             threshold=drop_score,
#             font_path=font_path)
#         img = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
#         return img
#     return image