import os
from config import TRAIN_DIR, TEST_DIR, IMAGE_DIR, VELODYNE_DIR, LABEL_DIR, CALIB_DIR
import cv2
import numpy as np

def cam2lidar(x, y, z, Tr_velo_to_cam):
    # 4x4 齐次矩阵
    Tr = np.vstack([Tr_velo_to_cam, [0,0,0,1]])
    Tr_inv = np.linalg.inv(Tr)
    point_cam = np.array([x, y, z, 1])
    point_lidar = Tr_inv @ point_cam
    return point_lidar[:3]

def parse_calib(calib_lines):
    # 提取 Tr_velo_to_cam
    Tr_line = [l for l in calib_lines if l.startswith('Tr_velo_to_cam')][0]
    values = list(map(float, Tr_line.strip().split()[1:]))
    Tr = np.array(values).reshape(3,4)
    return Tr

class DRadDataset:
    def __init__(self, split='train'):
        if split == 'train':
            self.root_dir = TRAIN_DIR
        elif split == 'test':
            self.root_dir = TEST_DIR
        else:
            raise ValueError("split must be 'train' or 'test'")

        self.image_dir = os.path.join(self.root_dir, IMAGE_DIR)
        self.velo_dir = os.path.join(self.root_dir, VELODYNE_DIR)
        self.label_dir = os.path.join(self.root_dir, LABEL_DIR)
        self.calib_dir = os.path.join(self.root_dir, CALIB_DIR)

        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".png")])
        self.velo_files = sorted([f for f in os.listdir(self.velo_dir) if f.endswith(".bin")])
        self.label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith(".txt")])
        self.calib_files = sorted([f for f in os.listdir(self.calib_dir) if f.endswith(".txt")])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 路径
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        velo_path  = os.path.join(self.velo_dir, self.velo_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        calib_path = os.path.join(self.calib_dir, self.calib_files[idx])

        # 1. 图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. 点云
        points = np.fromfile(velo_path, dtype=np.float32).reshape(-1,4)

        # 3. 标注
        with open(label_path, 'r') as f:
            labels_raw = f.readlines()

        # 4. 标定
        with open(calib_path, 'r') as f:
            calib_lines = f.readlines()
        Tr_velo_to_cam = parse_calib(calib_lines)

        # -----------------------
        # 5. 转换 bbox 到 LiDAR
        labels_lidar = []
        for line in labels_raw:
            parts = line.strip().split()
            cls = parts[0]
            # bbox 中心点
            x_cam, y_cam, z_cam = float(parts[11]), float(parts[12]), float(parts[13])
            x_lidar, y_lidar, z_lidar = cam2lidar(x_cam, y_cam, z_cam, Tr_velo_to_cam)
            # 更新中心点
            parts[11], parts[12], parts[13] = x_lidar, y_lidar, z_lidar
            labels_lidar.append(" ".join(map(str, parts)))

        return {
            "image": image,
            "points": points,
            "labels": labels_lidar,   # 已经是 LiDAR 坐标系
            "calib": calib_lines,
            "image_path": image_path,
            "velo_path": velo_path,
            "label_path": label_path,
            "calib_path": calib_path
        }
