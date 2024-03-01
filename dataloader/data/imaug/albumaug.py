import cv2
import albumentations as A
import numpy as np

class Albumentation(object):
    def __init__(self, **kwargs):
        self.transform = A.Compose([
        A.Blur(p = 0.1),
        A.MotionBlur(p = 0.1),
        A.GaussNoise(p = 0.1),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p = 0.2),
        A.CLAHE(p = 0.2),
        A.RandomBrightnessContrast(p=0.7),
        A.ISONoise(p = 0.3),
        A.RGBShift(p = 0.5),
        A.ChannelShuffle(p = 0.5),
        A.ShiftScaleRotate( shift_limit = 0.0, shift_limit_y = 0.01, scale_limit=0.0, rotate_limit=1, p=0.2),
        # A.ElasticTransform(alpha_affine=0.5, alpha=0.5, sigma=0, p = 0.1),
        # A.Perspective(p = 0.2),
        A.ToGray(p = 0.25),
        ])
    
    def __call__(self, data):
        img = data['image']
        transformed = self.transform(image=img)
        transformed_image = transformed['image']

        data['image'] = transformed_image
        return data
