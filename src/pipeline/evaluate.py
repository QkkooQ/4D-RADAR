# evaluate.py (Final Corrected Version)

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torchvision

from configs.default import DATASET_ROOT
from src.data.drad_dataset import DRadDataset
from src.models.pointpillars import PointPillars

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
            'num_anchors': 6
        }
    },
    'evaluation': {
        'batch_size': 30,
        'split': 'test',
        'output_dir': 'eval_results-2nd',
        'model_path': 'pointpillars_baseline.pth',
        'score_threshold': 0.3,
        'nms_iou_threshold': 0.1
    },
    'anchors': {
        'configs': [
            {'size': [3.9, 1.6, 1.56], 'rotation': [0, np.pi/2], 'height': -1.78, 'class': 'Car'},
            {'size': [1.76, 0.6, 1.73], 'rotation': [0, np.pi/2], 'height': -0.6,  'class': 'Cyclist'},
            {'size': [6.93, 2.51, 2.84], 'rotation': [0, np.pi/2], 'height': -1.78, 'class': 'Truck'},
        ]
    }
}

# --- 2. Re-used components from train.py ---
class AnchorGenerator:
    def __init__(self, anchor_configs): self.anchor_configs = anchor_configs
    @torch.no_grad()
    def generate(self, feature_map_size):
        x_centers = torch.linspace(CONFIG['model']['point_cloud_range'][0], CONFIG['model']['point_cloud_range'][3], steps=feature_map_size[1])
        y_centers = torch.linspace(CONFIG['model']['point_cloud_range'][1], CONFIG['model']['point_cloud_range'][4], steps=feature_map_size[0])
        cx, cy = torch.meshgrid(y_centers, x_centers, indexing='ij') # Use 'ij' indexing for consistency
        all_anchors = []
        for config in self.anchor_configs:
            for rot in config['rotation']:
                anchors_rot = torch.zeros((cx.shape[0], cx.shape[1], 7))
                anchors_rot[:, :, 0], anchors_rot[:, :, 1], anchors_rot[:, :, 2] = cx, cy, config['height']
                anchors_rot[:, :, 3:6] = torch.tensor(config['size'])
                anchors_rot[:, :, 6] = rot
                all_anchors.append(anchors_rot.reshape(-1, 7))
        return torch.cat(all_anchors, axis=0).float()

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

def eval_collate_fn(batch):
    points_list = [item['points'] for item in batch]
    batch_info = [{'frame_id': item['frame_id'], 'calib': item['calib']} for item in batch]
    pillar_features_list, pillar_coords_list = [], []
    for i, points in enumerate(points_list):
        features, coords = cpu_voxelize(points, **CONFIG['model'])
        if features.shape[0] == 0: continue
        pillar_features_list.append(torch.from_numpy(features))
        batch_idx_col = torch.full((features.shape[0], 1), i)
        coords_with_batch = torch.cat([batch_idx_col, torch.from_numpy(coords[:, ::-1].copy()).int()], dim=1)
        pillar_coords_list.append(coords_with_batch)
    if not pillar_features_list: return None
    return {'pillar_features': torch.cat(pillar_features_list, dim=0), 'pillar_coords': torch.cat(pillar_coords_list, dim=0), 'batch_info': batch_info}

# --- 3. Post-processing and Coordinate Transformation ---
@torch.no_grad()
def post_process(predictions, anchors, score_thresh, nms_thresh):
    """Decodes model output, applies NMS, and returns final detections."""
    cls_preds = torch.sigmoid(predictions['cls_preds'])
    box_preds = predictions['box_preds']
    
    decoded_boxes = torch.zeros_like(box_preds)
    decoded_boxes[:, :3] = box_preds[:, :3] * anchors[:, 3:6] + anchors[:, :3]
    decoded_boxes[:, 3:6] = torch.exp(box_preds[:, 3:6]) * anchors[:, 3:6]
    decoded_boxes[:, 6] = box_preds[:, 6] + anchors[:, 6]
    
    scores, labels = torch.max(cls_preds, dim=1)
    
    keep_mask = scores > score_thresh
    boxes_filtered = decoded_boxes[keep_mask]
    scores_filtered = scores[keep_mask]
    labels_filtered = labels[keep_mask]
    
    if boxes_filtered.shape[0] == 0:
        return [], [], []

    boxes_for_nms = boxes_filtered[:, [0, 1, 3, 4]]
    boxes_for_nms_xyxy = torch.zeros_like(boxes_for_nms)
    boxes_for_nms_xyxy[:, 0] = boxes_for_nms[:, 0] - boxes_for_nms[:, 2] / 2
    boxes_for_nms_xyxy[:, 1] = boxes_for_nms[:, 1] - boxes_for_nms[:, 3] / 2
    boxes_for_nms_xyxy[:, 2] = boxes_for_nms[:, 0] + boxes_for_nms[:, 2] / 2
    boxes_for_nms_xyxy[:, 3] = boxes_for_nms[:, 1] + boxes_for_nms[:, 3] / 2
    
    # --- CRITICAL FIX: Move tensors to CPU before calling NMS ---
    boxes_for_nms_cpu = boxes_for_nms_xyxy.cpu()
    scores_filtered_cpu = scores_filtered.cpu()
    
    nms_indices = torchvision.ops.nms(boxes_for_nms_cpu, scores_filtered_cpu, nms_thresh)
    # -----------------------------------------------------------
    
    final_boxes = boxes_filtered[nms_indices]
    final_scores = scores_filtered[nms_indices]
    final_labels = labels_filtered[nms_indices]
    
    return final_boxes, final_scores, final_labels

def transform_lidar_to_camera(boxes_lidar, calib):
    tr_velo_to_cam = torch.from_numpy(calib['Tr_velo_to_cam']).float().to(boxes_lidar.device)
    loc_lidar = boxes_lidar[:, :3]
    loc_lidar_h = torch.cat([loc_lidar, torch.ones(loc_lidar.shape[0], 1, device=loc_lidar.device)], dim=1)
    loc_cam = (loc_lidar_h @ tr_velo_to_cam.T)[:, :3]
    
    heading = boxes_lidar[:, 6]
    ry_cam = -heading - np.pi / 2
    
    dims_hwl = boxes_lidar[:, [5, 4, 3]]
    
    return torch.cat([dims_hwl, loc_cam, ry_cam.unsqueeze(-1)], dim=1)

def format_kitti_output(boxes_cam, scores, labels, class_names):
    output_lines = []
    for i in range(boxes_cam.shape[0]):
        box, score, label_idx = boxes_cam[i], scores[i], labels[i]
        
        h, w, l = box[0].item(), box[1].item(), box[2].item()
        x, y, z = box[3].item(), box[4].item(), box[5].item()
        ry = box[6].item()
        
        line = f"{class_names[label_idx]} 0.00 0 -10.00 0.00 0.00 0.00 0.00 " \
               f"{h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {ry:.2f} {score:.2f}"
        output_lines.append(line)
    return output_lines

# --- 4. Main Evaluation Loop ---
def main():
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_config = CONFIG['evaluation']
    
    model = PointPillars(CONFIG['model']).to(device)
    # Add weights_only=True for safer loading, suppressing the warning
    model.load_state_dict(torch.load(eval_config['model_path'], weights_only=True))
    model.eval()
    print(f"Model loaded from {eval_config['model_path']}")

    dataset = DRadDataset(root_path=CONFIG['dataset']['root_path'], split=eval_config['split'])
    dataloader = DataLoader(dataset, batch_size=eval_config['batch_size'], shuffle=False, num_workers=0, collate_fn=eval_collate_fn)
    
    feature_map_size = (int(CONFIG['model']['scatter']['output_shape'][0] / 4), int(CONFIG['model']['scatter']['output_shape'][1] / 4))
    anchor_generator = AnchorGenerator(CONFIG['anchors']['configs'])
    anchors = anchor_generator.generate(feature_map_size).to(device)

    os.makedirs(eval_config['output_dir'], exist_ok=True)
    
    print(f"--- Starting Evaluation on '{eval_config['split']}' split ---")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch is None: continue
            
            pillar_features = batch['pillar_features'].to(device)
            pillar_coords = batch['pillar_coords'].to(device)
            
            predictions = model(pillar_features, pillar_coords)
            
            cls_preds = predictions['cls_preds'].permute(0, 2, 3, 1).contiguous().view(len(batch['batch_info']), -1, CONFIG['model']['head']['num_classes'])
            box_preds = predictions['box_preds'].permute(0, 2, 3, 1).contiguous().view(len(batch['batch_info']), -1, 7)
            
            for i in range(len(batch['batch_info'])):
                frame_id = batch['batch_info'][i]['frame_id']
                calib = batch['batch_info'][i]['calib']

                final_boxes, final_scores, final_labels = post_process(
                    {'cls_preds': cls_preds[i], 'box_preds': box_preds[i]},
                    anchors,
                    eval_config['score_threshold'],
                    eval_config['nms_iou_threshold']
                )

                if len(final_boxes) == 0:
                    output_path = os.path.join(eval_config['output_dir'], f"{frame_id}.txt")
                    open(output_path, 'w').close()
                    continue

                boxes_cam = transform_lidar_to_camera(final_boxes, calib)
                output_lines = format_kitti_output(boxes_cam.cpu(), final_scores.cpu(), final_labels.cpu(), CONFIG['dataset']['class_names'])
                
                output_path = os.path.join(eval_config['output_dir'], f"{frame_id}.txt")
                with open(output_path, 'w') as f:
                    f.write('\n'.join(output_lines))

    print(f"\n--- Evaluation Finished ---")
    print(f"Prediction files saved to: '{eval_config['output_dir']}'")

if __name__ == "__main__":
    main()
