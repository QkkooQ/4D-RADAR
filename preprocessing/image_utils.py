# preprocessing/image_utils.py
import cv2
import numpy as np

def load_image(path):
    """
    读取图像
    返回BGR格式的numpy数组
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at {path}")
    return img

def resize_image(img, target_size=(640, 480)):
    """
    调整图像大小
    target_size: (width, height)
    """
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    return resized

def normalize_image(img):
    """
    将图像归一化到0~1，并转换为float32
    """
    img = img.astype(np.float32) / 255.0
    return img

def preprocess_image(path, target_size=(640,480)):
    """
    综合处理：读取 -> resize -> normalize
    """
    img = load_image(path)
    img = resize_image(img, target_size)
    img = normalize_image(img)
    return img
