import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def get_bbox_points(h, w, l, x, y, z, ry):
    """
    返回 8 个角点 (x, y, z)
    h: 高, w: 宽, l: 长, ry: 绕 Y 旋转
    """
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners = np.vstack([x_corners, y_corners, z_corners])
    R = np.array([[ np.cos(ry), 0, np.sin(ry)],
                  [ 0,          1, 0        ],
                  [-np.sin(ry), 0, np.cos(ry)]])
    corners_rot = np.dot(R, corners)
    corners_rot[0,:] += x
    corners_rot[1,:] += y
    corners_rot[2,:] += z
    return corners_rot.T  # 8x3

def draw_pointcloud(points, labels=None, class_colors=None):
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Point Cloud")
    xs, ys, zs = points[:,0], points[:,1], points[:,2]
    
    # 默认灰色
    color_list = np.array([[0.5,0.5,0.5]]*len(points))
    
    if labels is not None and class_colors is not None:
        for label_line in labels:
            parts = label_line.strip().split()
            cls = parts[0]
            h, w, l = float(parts[10]), float(parts[8]), float(parts[9])
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            ry = float(parts[14])
            
            # 计算 8 个角点
            corners = get_bbox_points(h, w, l, x, y, z, ry)
            
            # AABB 包围盒简单上色点
            x_min, x_max = corners[:,0].min(), corners[:,0].max()
            y_min, y_max = corners[:,1].min(), corners[:,1].max()
            z_min, z_max = corners[:,2].min(), corners[:,2].max()
            mask = (points[:,0]>=x_min) & (points[:,0]<=x_max) & \
                   (points[:,1]>=y_min) & (points[:,1]<=y_max) & \
                   (points[:,2]>=z_min) & (points[:,2]<=z_max)
            color = np.array(class_colors.get(cls,(128,128,128)))/255.0
            color_list[mask] = color

    ax.scatter(xs, ys, zs, c=color_list, s=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
