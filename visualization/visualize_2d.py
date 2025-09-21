import cv2
import numpy as np

def parse_bbox(label_line):
    """
    返回类别和 2D bbox
    """
    parts = label_line.strip().split()
    cls = parts[0]
    bbox = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]  # left, top, right, bottom
    return cls, bbox

def draw_image_with_bbox(image, labels, class_colors):
    img = image.copy()
    for label_line in labels:
        cls, bbox = parse_bbox(label_line)
        color = class_colors.get(cls, (255, 255, 255))
        bbox_int = [int(x) for x in bbox]
        cv2.rectangle(img, (bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3]), color, 2)
        cv2.putText(img, cls, (bbox_int[0], bbox_int[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img
