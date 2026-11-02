# datasets/kitti_utils.py
import numpy as np

def read_label_file(label_path):
    """
    读取 KITTI 格式的 2D/3D 标注
    返回每个物体的 dict 列表
    """
    objects = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(' ')
            obj = {
                "class": parts[0],
                "truncated": float(parts[1]),
                "occluded": int(parts[2]),
                "alpha": float(parts[3]),
                "bbox": [float(x) for x in parts[4:8]],        # 2D bbox: xmin, ymin, xmax, ymax
                "dimensions": [float(x) for x in parts[8:11]], # 高宽长 h,w,l
                "location": [float(x) for x in parts[11:14]],  # 3D位置 x,y,z
                "rotation_y": float(parts[14])                 # 旋转角度
            }
            objects.append(obj)
    return objects

def read_calib_file(calib_path):
    """
    读取标定文件，返回字典
    P2: 相机投影矩阵
    Tr_velo_to_cam: 激光到相机坐标转换矩阵
    """
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            if line.startswith("P2:"):
                calib["P2"] = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3,4)
            elif line.startswith("Tr_velo_to_cam:"):
                calib["Tr_velo_to_cam"] = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3,4)
    return calib
