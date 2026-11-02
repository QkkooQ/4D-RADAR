import argparse
import matplotlib.pyplot as plt
import numpy as np

from configs.default import CLASS_COLORS, DATASET_ROOT
from src.data.drad_dataset import DRadDataset
from src.visualization.visualize_2d import draw_image_with_bbox
from src.visualization.visualize_3d import get_bbox_points

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
    dataset = DRadDataset(root_path=DATASET_ROOT, split='train')
    print(f"数据集大小: {len(dataset)}")
    
    num_frames = min(args.num_frames, len(dataset))
    print(f"Demo for {num_frames} frames...")

    for i in range(num_frames):
        data = dataset[i]
        image = data['image']
        points = data['points']
        gt_boxes = data.get('gt_boxes', np.empty((0, 7)))
        gt_names = data.get('gt_names', [])

        fig = plt.figure(figsize=(12,6))

        # 左 2D
        ax1 = fig.add_subplot(121)
        img_with_bbox = image
        ax1.imshow(img_with_bbox)
        ax1.set_title(f"Frame {i} 2D")
        ax1.axis('off')

        # 右 3D
        ax2 = fig.add_subplot(122, projection='3d')
        xs, ys, zs = points[:,0], points[:,1], points[:,2]
        ax2.scatter(xs, ys, zs, c='gray', s=1)

        for box, class_name in zip(gt_boxes, gt_names):
            x, y, z, length, width, height, heading = box
            corners = get_bbox_points(height, width, length, x, y, z, heading)
            color = np.array(CLASS_COLORS.get(class_name, (255, 255, 255))) / 255.0
            draw_bbox_lines(ax2, corners, color=color)

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
