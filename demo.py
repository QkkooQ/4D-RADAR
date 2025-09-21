import argparse
from datasets.drad_dataset import DRadDataset
import config
from visualization import visualize_2d
from visualization.visualize_3d import get_bbox_points
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

def draw_bbox_lines(ax, corners, color='r'):
    """
    在 ax 上画 3D bbox 的边线
    corners: 8x3
    """
    # 所有角点组合，距离为边长的连线
    for i, j in [
        (0,1),(1,2),(2,3),(3,0), # 底面
        (4,5),(5,6),(6,7),(7,4), # 顶面
        (0,4),(1,5),(2,6),(3,7)  # 垂直边
    ]:
        xs, ys, zs = [corners[i,0], corners[j,0]], [corners[i,1], corners[j,1]], [corners[i,2], corners[j,2]]
        ax.plot(xs, ys, zs, color=color, linewidth=1)

def main(args):
    dataset = DRadDataset(split='train')
    print(f"数据集大小: {len(dataset)}")
    
    num_frames = min(args.num_frames, len(dataset))
    print(f"Demo for {num_frames} frames...")

    for i in range(num_frames):
        data = dataset[i]
        image = data['image']
        points = data['points']
        labels = data['labels']

        fig = plt.figure(figsize=(12,6))

        # 左 2D
        ax1 = fig.add_subplot(121)
        img_with_bbox = visualize_2d.draw_image_with_bbox(image, labels, config.CLASS_COLORS)
        ax1.imshow(img_with_bbox)
        ax1.set_title(f"Frame {i} 2D")
        ax1.axis('off')

        # 右 3D
        ax2 = fig.add_subplot(122, projection='3d')
        xs, ys, zs = points[:,0], points[:,1], points[:,2]
        ax2.scatter(xs, ys, zs, c='gray', s=1)

        if labels is not None:
            for label_line in labels:
                parts = label_line.strip().split()
                cls = parts[0]
                if len(parts) < 15:
                    continue  # 没有 3D 信息
                try:
                    h, w, l = float(parts[10]), float(parts[8]), float(parts[9])
                    x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
                    ry = float(parts[14])
                except:
                    continue

                corners = get_bbox_points(h, w, l, x, y, z, ry)
                print(f"Frame {i}, 类别 {cls}, bbox corners:\n", corners)

                draw_bbox_lines(ax2, corners, color='r')

        ax2.set_title(f"Frame {i} 3D")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        plt.tight_layout()
        plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_frames', type=int, default=5)
    args = parser.parse_args()
    main(args)
