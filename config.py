# config.py
import os

# 根目录，根据你的实际存放位置修改
ROOT_DIR = r"C:/Users/kkoo1/Desktop/4D_LiDar/4D_PROJ/DRadDataset"

# 训练/测试数据路径
TRAIN_DIR = os.path.join(ROOT_DIR, "training")
TEST_DIR  = os.path.join(ROOT_DIR, "testing")

# 数据子目录
IMAGE_DIR = "image_2"
VELODYNE_DIR = "velodyne"
CALIB_DIR = "calib"
LABEL_DIR = "label_2"

# 类别映射（根据比赛提供的类别调整）
CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]

# 可视化颜色映射
CLASS_COLORS = {
    "Car": (255, 0, 0),         # 红
    "Pedestrian": (0, 255, 0),  # 绿
    "Cyclist": (0, 0, 255),     # 蓝
}
