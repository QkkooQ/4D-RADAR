# train.py (Final Version)

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from functools import partial

from configs.default import DATASET_ROOT
from src.data.drad_dataset import DRadDataset
from src.models.pointpillars import PointPillars
from src.models.losses import PointPillarsLoss

# --- 1. Configuration ---
CONFIG = {
    'dataset': {
        'root_path': str(DATASET_ROOT),
        'class_names': ['Car', 'Cyclist', 'Truck']
    },
    'model': {
        'point_cloud_range': [0, -25.6, -3, 51.2, 25.6, 1],
        'voxel_size': [0.16, 0.16, 4],
        'max_points_per_voxel': 32,
        'max_voxels': 12000,
        'pfn': { 'num_input_features': 5, 'num_filters': [64], 'out_channels': 64 },
        'scatter': { 'output_shape': (320, 320), 'num_bev_features': 64 },
        'backbone': { 'in_channels': 64, 'out_channels': 128 },
        'head': {
            'in_channels': 128,
            'num_classes': 3,
            'num_anchors': 6 # 3 classes * 2 rotations
        }
    },
    'training': {
        'batch_size': 25, # 可以根据你的GPU显存适当调大
        'num_epochs': 50,
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    },
    'anchors': {
        'configs': [
            {'size': [3.9, 1.6, 1.56], 'rotation': [0, np.pi/2], 'height': -1.78}, # Car
            {'size': [1.76, 0.6, 1.73], 'rotation': [0, np.pi/2], 'height': -0.6},  # Cyclist
            {'size': [6.93, 2.51, 2.84], 'rotation': [0, np.pi/2], 'height': -1.78}, # Truck
        ],
        'pos_iou_thresh': 2.0, # Using distance as proxy, so threshold is in meters
        'neg_iou_thresh': 3.5
    }
}

# --- 2. Anchor Generation and Target Assignment ---
class AnchorGenerator:
    def __init__(self, anchor_configs):
        self.anchor_configs = anchor_configs

    @torch.no_grad()
    def generate(self, feature_map_size):
        # Generate anchors based on the output feature map size
        x_centers = torch.linspace(CONFIG['model']['point_cloud_range'][0], CONFIG['model']['point_cloud_range'][3], steps=feature_map_size[1])
        y_centers = torch.linspace(CONFIG['model']['point_cloud_range'][1], CONFIG['model']['point_cloud_range'][4], steps=feature_map_size[0])
        cx, cy = torch.meshgrid(y_centers, x_centers)

        all_anchors = []
        for config in self.anchor_configs:
            for rot in config['rotation']:
                anchors_rot = torch.zeros((cx.shape[0], cx.shape[1], 7))
                anchors_rot[:, :, 0] = cx
                anchors_rot[:, :, 1] = cy
                anchors_rot[:, :, 2] = config['height']
                anchors_rot[:, :, 3:6] = torch.tensor(config['size'])
                anchors_rot[:, :, 6] = rot
                all_anchors.append(anchors_rot.reshape(-1, 7))
        return torch.cat(all_anchors, axis=0).float()

class TargetAssigner:
    def __init__(self, anchor_generator, pos_thresh, neg_thresh):
        self.anchor_generator = anchor_generator
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh

    @torch.no_grad()
    def assign(self, anchors, gt_boxes, gt_classes):
        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]
        
        cls_labels = torch.full((num_anchors,), 0, dtype=torch.long) # Default to background
        reg_labels = torch.zeros((num_anchors, 7))

        if num_gt == 0:
            return cls_labels, reg_labels

        # Using distance-based matching as a simplified proxy for IoU
        anchor_centers = anchors[:, :2]
        gt_centers = gt_boxes[:, :2]
        dist_matrix = torch.cdist(anchor_centers, gt_centers)
        
        # For each anchor, find the closest GT box
        min_gt_dist, min_gt_indices = torch.min(dist_matrix, dim=1)
        
        # Assign positive anchors (anchors that are close enough to ANY GT box)
        pos_mask = min_gt_dist < self.pos_thresh
        if pos_mask.sum() > 0:
            matched_gt_boxes = gt_boxes[min_gt_indices[pos_mask]]
            matched_gt_classes = gt_classes[min_gt_indices[pos_mask]]
            
            # Encode GT boxes relative to the anchor
            pos_anchors = anchors[pos_mask]
            reg_target = torch.zeros_like(pos_anchors)
            reg_target[:, :3] = (matched_gt_boxes[:, :3] - pos_anchors[:, :3]) / pos_anchors[:, 3:6]
            reg_target[:, 3:6] = torch.log(matched_gt_boxes[:, 3:6] / pos_anchors[:, 3:6])
            reg_target[:, 6] = matched_gt_boxes[:, 6] - pos_anchors[:, 6]
            
            reg_labels[pos_mask] = reg_target
            # Class labels should be 1-based (0 is background)
            cls_labels[pos_mask] = matched_gt_classes + 1

        return cls_labels, reg_labels

# --- 3. Voxelization (Unchanged) & Collate Function ---
def cpu_voxelize(points, **kwargs):
    # This function is unchanged. Please use the corrected version from our previous discussion.
    voxel_size = kwargs['voxel_size']
    point_cloud_range = kwargs['point_cloud_range']
    max_points_per_voxel = kwargs['max_points_per_voxel']
    max_voxels = kwargs['max_voxels']
    # ... (paste the full corrected cpu_voxelize function here) ...
    mask = np.all((points[:, :3] >= point_cloud_range[:3]) & (points[:, :3] <= point_cloud_range[3:]), axis=1)
    points = points[mask]
    if points.shape[0] == 0: return np.zeros((0, max_points_per_voxel, 5)), np.zeros((0, 3))
    voxel_coords = np.floor((points[:, :3] - np.array(point_cloud_range[:3])) / np.array(voxel_size)).astype(int)
    voxel_coords_str, point_indices, counts = np.unique(['_'.join(map(str, c)) for c in voxel_coords], return_inverse=True, return_counts=True)
    num_pillars = len(voxel_coords_str)
    if num_pillars > max_voxels:
        pillar_indices_to_keep = np.random.choice(np.arange(num_pillars), max_voxels, replace=False)
    else:
        pillar_indices_to_keep = np.arange(num_pillars)
    
    pillar_features = np.zeros((len(pillar_indices_to_keep), max_points_per_voxel, points.shape[1]), dtype=np.float32)
    final_coords_int = np.array([list(map(int, s.split('_'))) for s in voxel_coords_str[pillar_indices_to_keep]], dtype=np.int32)
    
    for i, pillar_idx in enumerate(pillar_indices_to_keep):
        points_in_pillar = points[point_indices == pillar_idx]
        num_points = points_in_pillar.shape[0]
        if num_points >= max_points_per_voxel:
            sampled_indices = np.random.choice(num_points, max_points_per_voxel, replace=False)
            pillar_features[i, :, :] = points_in_pillar[sampled_indices]
        else:
            pillar_features[i, :num_points, :] = points_in_pillar
    return pillar_features, final_coords_int


def custom_collate_fn(batch, anchor_generator, target_assigner, feature_map_size, class_names):
    points_list = [item['points'] for item in batch]
    gt_boxes_list = [torch.from_numpy(item['gt_boxes']).float() for item in batch]
    gt_names_list = [item['gt_names'] for item in batch]

    pillar_features_list, pillar_coords_list = [], []
    cls_labels_list, reg_labels_list = [], []

    anchors = anchor_generator.generate(feature_map_size=feature_map_size)
    class_map = {name: i for i, name in enumerate(class_names)}
    
    for i, (points, gt_boxes, gt_names) in enumerate(zip(points_list, gt_boxes_list, gt_names_list)):
        features, coords = cpu_voxelize(points, **CONFIG['model'])
        pillar_features_list.append(torch.from_numpy(features))
        batch_idx_col = torch.full((features.shape[0], 1), i)
        coords_with_batch = torch.cat([batch_idx_col, torch.from_numpy(coords[:, ::-1].copy())], dim=1)
        pillar_coords_list.append(coords_with_batch)
        
        gt_classes = torch.tensor([class_map[name] for name in gt_names])
        cls_labels, reg_labels = target_assigner.assign(anchors, gt_boxes, gt_classes)
        cls_labels_list.append(cls_labels)
        reg_labels_list.append(reg_labels)

    return {
        'pillar_features': torch.cat(pillar_features_list, dim=0),
        'pillar_coords': torch.cat(pillar_coords_list, dim=0),
        'cls_labels': torch.stack(cls_labels_list, dim=0),
        'reg_labels': torch.stack(reg_labels_list, dim=0)
    }

# --- 4. Main Training Loop ---
def main():
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    device = torch.device(CONFIG['training']['device'])
    print(f"Using device: {device}")
    
    feature_map_size = (
        int(CONFIG['model']['scatter']['output_shape'][0] / 4),
        int(CONFIG['model']['scatter']['output_shape'][1] / 4)
    )
    
    anchor_generator = AnchorGenerator(CONFIG['anchors']['configs'])
    target_assigner = TargetAssigner(anchor_generator, pos_thresh=CONFIG['anchors']['pos_iou_thresh'], neg_thresh=CONFIG['anchors']['neg_iou_thresh'])
    
    dataset = DRadDataset(root_path=CONFIG['dataset']['root_path'], split='train')
    collate_fn = partial(custom_collate_fn, anchor_generator=anchor_generator, target_assigner=target_assigner, feature_map_size=feature_map_size, class_names=CONFIG['dataset']['class_names'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['training']['batch_size'], shuffle=True, num_workers=0, collate_fn=collate_fn)

    model = PointPillars(CONFIG['model']).to(device)
    loss_func = PointPillarsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['training']['learning_rate'])

    print("--- Starting Training with Real Labels and Loss ---")
    for epoch in range(CONFIG['training']['num_epochs']):
        total_cls_loss, total_reg_loss = 0, 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['training']['num_epochs']}")
        
        for batch in pbar:
            if batch is None:
                continue
            pillar_features = batch['pillar_features'].to(device)
            pillar_coords = batch['pillar_coords'].to(device)
            cls_labels = batch['cls_labels'].to(device)
            reg_labels = batch['reg_labels'].to(device)
            
            predictions = model(pillar_features, pillar_coords)
            
            cls_preds = predictions['cls_preds'].permute(0, 2, 3, 1).contiguous().view(cls_labels.shape[0], -1, CONFIG['model']['head']['num_classes'])
            box_preds = predictions['box_preds'].permute(0, 2, 3, 1).contiguous().view(reg_labels.shape[0], -1, 7)
            
            loss_dict = loss_func(cls_preds, box_preds, cls_labels, reg_labels)
            loss = loss_dict['total_loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_cls_loss += loss_dict['cls_loss'].item()
            total_reg_loss += loss_dict['reg_loss'].item()
            pbar.set_postfix(cls=loss_dict['cls_loss'].item(), reg=loss_dict['reg_loss'].item())

        avg_cls = total_cls_loss / len(dataloader)
        avg_reg = total_reg_loss / len(dataloader)
        print(f"Epoch {epoch+1} >> Avg Cls Loss: {avg_cls:.4f}, Avg Reg Loss: {avg_reg:.4f}")

    torch.save(model.state_dict(), 'pointpillars_baseline.pth')
    print("--- Training Finished --- \nModel saved to pointpillars_baseline.pth")

if __name__ == "__main__":
    main()
