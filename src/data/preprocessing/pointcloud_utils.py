# preprocessing/pointcloud_utils.py
import numpy as np

def crop_pointcloud(pointcloud, x_range=(-50, 50), y_range=(-50, 50), z_range=(-5, 5)):
    """
    裁剪点云到指定范围
    pointcloud: np.array (N,4) -> x,y,z,intensity
    返回裁剪后的点云
    """
    mask = (
        (pointcloud[:,0] >= x_range[0]) & (pointcloud[:,0] <= x_range[1]) &
        (pointcloud[:,1] >= y_range[0]) & (pointcloud[:,1] <= y_range[1]) &
        (pointcloud[:,2] >= z_range[0]) & (pointcloud[:,2] <= z_range[1])
    )
    return pointcloud[mask]

def downsample_pointcloud(pointcloud, max_points=100000):
    """
    随机下采样点云到指定数量
    """
    if len(pointcloud) <= max_points:
        return pointcloud
    else:
        idx = np.random.choice(len(pointcloud), max_points, replace=False)
        return pointcloud[idx]

def normalize_pointcloud(pointcloud):
    """
    对点云进行简单归一化（每个维度减去均值再除以标准差）
    intensity单独归一化到0~1
    """
    pc = pointcloud.copy()
    xyz = pc[:,:3]
    intensity = pc[:,3:4]

    # 均值方差归一化
    mean = xyz.mean(axis=0)
    std = xyz.std(axis=0)
    xyz = (xyz - mean) / (std + 1e-6)

    # intensity归一化到0~1
    intensity = intensity / (intensity.max() + 1e-6)

    return np.hstack([xyz, intensity])

def preprocess_pointcloud(pointcloud, x_range=(-50,50), y_range=(-50,50), z_range=(-5,5), max_points=100000):
    """
    综合预处理函数：裁剪 -> 下采样 -> 归一化
    """
    pc = crop_pointcloud(pointcloud, x_range, y_range, z_range)
    pc = downsample_pointcloud(pc, max_points)
    pc = normalize_pointcloud(pc)
    return pc
