# train.py (Final Fusion Version)

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from functools import partial

from configs.default import DATASET_ROOT
from src.data.drad_dataset import DRadDataset
from src.models.fusion import PointPillarsFusion
from src.models.losses import PointPillarsLoss

# --- 1. Configuration (Updated for Fusion) ---
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
        'image_encoder': { 'out_channels': 64 },
        'backbone': { 'in_channels': 128, 'out_channels': 128 }, # 64 (lidar) + 64 (image)
        'head': { 'in_channels': 128, 'num_classes': 3, 'num_anchors': 6 }
    },
    'training': {
        'batch_size': 8,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    },
    'anchors': {
        'configs': [
            {'size': [3.9, 1.6, 1.56], 'rotation': [0, np.pi/2], 'height': -1.78}, # Car
            {'size': [1.76, 0.6, 1.73], 'rotation': [0, np.pi/2], 'height': -0.6},  # Cyclist
            {'size': [6.93, 2.51, 2.84], 'rotation': [0, np.pi/2], 'height': -1.78}, # Truck
        ],
        'pos_iou_thresh': 2.0, # Using distance as a proxy for IoU, threshold in meters
        'neg_iou_thresh': 3.5
    }
}

# --- 2. Anchor/Target Modules & Voxelization (UNCHANGED) ---
class AnchorGenerator:
    def __init__(self, anchor_configs): self.anchor_configs = anchor_configs
    @torch.no_grad()
    def generate(self, feature_map_size):
        x_centers = torch.linspace(CONFIG['model']['point_cloud_range'][0], CONFIG['model']['point_cloud_range'][3], steps=feature_map_size[1])
        y_centers = torch.linspace(CONFIG['model']['point_cloud_range'][1], CONFIG['model']['point_cloud_range'][4], steps=feature_map_size[0])
        cx, cy = torch.meshgrid(y_centers, x_centers, indexing='ij')
        all_anchors = []
        for config in self.anchor_configs:
            for rot in config['rotation']:
                anchors_rot = torch.zeros((cx.shape[0], cx.shape[1], 7))
                anchors_rot[:, :, 0], anchors_rot[:, :, 1], anchors_rot[:, :, 2] = cx, cy, config['height']
                anchors_rot[:, :, 3:6] = torch.tensor(config['size'])
                anchors_rot[:, :, 6] = rot
                all_anchors.append(anchors_rot.reshape(-1, 7))
        return torch.cat(all_anchors, axis=0).float()

class TargetAssigner:
    def __init__(self, anchor_generator, pos_thresh, neg_thresh):
        self.anchor_generator, self.pos_thresh, self.neg_thresh = anchor_generator, pos_thresh, neg_thresh
    @torch.no_grad()
    def assign(self, anchors, gt_boxes, gt_classes):
        num_anchors, num_gt = anchors.shape[0], gt_boxes.shape[0]
        cls_labels = torch.full((num_anchors,), 0, dtype=torch.long)
        reg_labels = torch.zeros((num_anchors, 7))
        if num_gt == 0: return cls_labels, reg_labels
        dist_matrix = torch.cdist(anchors[:, :2], gt_boxes[:, :2])
        min_gt_dist, min_gt_indices = torch.min(dist_matrix, dim=1)
        pos_mask = min_gt_dist < self.pos_thresh
        if pos_mask.sum() > 0:
            matched_gt_boxes, matched_gt_classes = gt_boxes[min_gt_indices[pos_mask]], gt_classes[min_gt_indices[pos_mask]]
            pos_anchors = anchors[pos_mask]
            reg_target = torch.zeros_like(pos_anchors)
            reg_target[:, :3] = (matched_gt_boxes[:, :3] - pos_anchors[:, :3]) / pos_anchors[:, 3:6]
            reg_target[:, 3:6] = torch.log(matched_gt_boxes[:, 3:6] / pos_anchors[:, 3:6])
            reg_target[:, 6] = matched_gt_boxes[:, 6] - pos_anchors[:, 6]
            reg_labels[pos_mask] = reg_target
            cls_labels[pos_mask] = matched_gt_classes + 1
        return cls_labels, reg_labels

def cpu_voxelize(points, **kwargs):
    voxel_size, point_cloud_range, max_points_per_voxel, max_voxels = kwargs['voxel_size'], kwargs['point_cloud_range'], kwargs['max_points_per_voxel'], kwargs['max_voxels']
    mask = np.all((points[:, :3] >= point_cloud_range[:3]) & (points[:, :3] <= point_cloud_range[3:]), axis=1)
    points = points[mask]
    if points.shape[0] == 0: return np.zeros((0, max_points_per_voxel, 5)), np.zeros((0, 3))
    voxel_coords = np.floor((points[:, :3] - np.array(point_cloud_range[:3])) / np.array(voxel_size)).astype(int)
    voxel_coords_str, point_indices = np.unique(['_'.join(map(str, c)) for c in voxel_coords], return_inverse=True)
    num_pillars = len(voxel_coords_str)
    if num_pillars > max_voxels: kept_pillar_indices = np.random.choice(np.arange(num_pillars), max_voxels, replace=False)
    else: kept_pillar_indices = np.arange(num_pillars)
    pillar_features = np.zeros((len(kept_pillar_indices), max_points_per_voxel, points.shape[1]), dtype=np.float32)
    final_coords_int = np.array([list(map(int, s.split('_'))) for s in voxel_coords_str[kept_pillar_indices]], dtype=np.int32)
    for i, pillar_idx in enumerate(kept_pillar_indices):
        points_in_pillar = points[point_indices == pillar_idx]
        num_points = points_in_pillar.shape[0]
        if num_points >= max_points_per_voxel:
            sampled_indices = np.random.choice(num_points, max_points_per_voxel, replace=False)
            pillar_features[i, :, :] = points_in_pillar[sampled_indices]
        else:
            pillar_features[i, :num_points, :] = points_in_pillar
    return pillar_features, final_coords_int

# --- 3. Custom Collate Function (Updated for Fusion) ---
def custom_collate_fn(batch, anchor_generator, target_assigner, feature_map_size, class_names):
    points_list = [item['points'] for item in batch]
    gt_boxes_list = [torch.from_numpy(item['gt_boxes']).float() for item in batch]
    gt_names_list = [item['gt_names'] for item in batch]
    image_list = [torch.from_numpy(item['image']).permute(2, 0, 1).float() / 255.0 for item in batch]
    calib_list = [item['calib'] for item in batch]

    pillar_features_list, pillar_coords_list, cls_labels_list, reg_labels_list = [], [], [], []
    anchors = anchor_generator.generate(feature_map_size=feature_map_size)
    class_map = {name: i for i, name in enumerate(class_names)}
    
    for i, (points, gt_boxes, gt_names) in enumerate(zip(points_list, gt_boxes_list, gt_names_list)):
        features, coords = cpu_voxelize(points, **CONFIG['model'])
        if features.shape[0] == 0: continue
        pillar_features_list.append(torch.from_numpy(features))
        batch_idx_col = torch.full((features.shape[0], 1), i, dtype=torch.int32)
        coords_with_batch = torch.cat([batch_idx_col, torch.from_numpy(coords[:, ::-1].copy()).int()], dim=1)
        pillar_coords_list.append(coords_with_batch)
        
        gt_classes = torch.tensor([class_map.get(name) for name in gt_names if name in class_map])
        valid_gt_mask = [name in class_map for name in gt_names]
        gt_boxes_filtered = gt_boxes[valid_gt_mask]

        cls_labels, reg_labels = target_assigner.assign(anchors, gt_boxes_filtered, gt_classes)
        cls_labels_list.append(cls_labels)
        reg_labels_list.append(reg_labels)

    if not pillar_features_list: return None
    return {
        'pillar_features': torch.cat(pillar_features_list, dim=0),
        'pillar_coords': torch.cat(pillar_coords_list, dim=0),
        'cls_labels': torch.stack(cls_labels_list, dim=0),
        'reg_labels': torch.stack(reg_labels_list, dim=0),
        'image': torch.stack(image_list, dim=0),
        'calib': calib_list
    }

# --- 4. Main Training Loop ---
def main():
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    device = torch.device(CONFIG['training']['device'])
    print(f"Using device: {device}")
    
    feature_map_size = (int(CONFIG['model']['scatter']['output_shape'][0] / 4), int(CONFIG['model']['scatter']['output_shape'][1] / 4))
    
    anchor_generator = AnchorGenerator(CONFIG['anchors']['configs'])
    target_assigner = TargetAssigner(anchor_generator, pos_thresh=CONFIG['anchors']['pos_iou_thresh'], neg_thresh=CONFIG['anchors']['neg_iou_thresh'])
    
    dataset = DRadDataset(root_path=CONFIG['dataset']['root_path'], split='train')
    collate_fn = partial(custom_collate_fn, anchor_generator=anchor_generator, target_assigner=target_assigner, feature_map_size=feature_map_size, class_names=CONFIG['dataset']['class_names'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['training']['batch_size'], shuffle=True, num_workers=0, collate_fn=collate_fn)

    model = PointPillarsFusion(CONFIG['model']).to(device)
    loss_func = PointPillarsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['training']['learning_rate'])

    print("--- Starting FUSION Model Training ---")
    for epoch in range(CONFIG['training']['num_epochs']):
        total_cls_loss, total_reg_loss = 0, 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['training']['num_epochs']}")
        
        for batch in pbar:
            if batch is None: continue
            
            for key in ['pillar_features', 'pillar_coords', 'cls_labels', 'reg_labels', 'image']:
                batch[key] = batch[key].to(device)
            
            predictions = model(batch)
            
            cls_preds = predictions['cls_preds'].permute(0, 2, 3, 1).contiguous().view(batch['cls_labels'].shape[0], -1, CONFIG['model']['head']['num_classes'])
            box_preds = predictions['box_preds'].permute(0, 2, 3, 1).contiguous().view(batch['reg_labels'].shape[0], -1, 7)
            
            loss_dict = loss_func(cls_preds, box_preds, batch['cls_labels'], batch['reg_labels'])
            loss = loss_dict['total_loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_cls_loss += loss_dict['cls_loss'].item()
            total_reg_loss += loss_dict['reg_loss'].item()
            pbar.set_postfix(cls=loss_dict['cls_loss'].item(), reg=loss_dict['reg_loss'].item())

        avg_cls = total_cls_loss / len(dataloader) if len(dataloader) > 0 else 0
        avg_reg = total_reg_loss / len(dataloader) if len(dataloader) > 0 else 0
        print(f"Epoch {epoch+1} >> Avg Cls Loss: {avg_cls:.4f}, Avg Reg Loss: {avg_reg:.4f}")

    torch.save(model.state_dict(), 'pointpillars_fusion_baseline.pth')
    print("--- Training Finished --- \nFusion model saved to pointpillars_fusion_baseline.pth")

if __name__ == "__main__":
    main()
