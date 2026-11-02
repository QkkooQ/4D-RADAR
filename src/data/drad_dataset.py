# drad_dataset.py

import os
import numpy as np
import cv2
from torch.utils.data import Dataset

class DRadDataset(Dataset):
    """
    DRadDataset PyTorch Dataset class.
    
    This class loads data based on the final, corrected understanding of the dataset format:
    1.  The .bin file is 8-dimensional, with the first 3 columns being x_file, y_file, z_file.
    2.  The key coordinate translation rule is applied: the file's (x_file, y_file, z_file)
        are interpreted as the camera system's (z_cam, -x_cam, -y_cam).
    3.  All 3D data (points and labels) is first brought into the common Camera Coordinate
        System and then transformed into the Lidar Coordinate System for model consumption.
    """
    def __init__(self, root_path, split='train'):
        super().__init__()
        self.root_path = root_path
        
        self.is_test = split == 'test'
        data_dir = os.path.join(self.root_path, 'testing' if self.is_test else 'training')
        
        split_file = os.path.join(self.root_path, 'ImageSets', f'{split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        self.sample_ids = [line.strip() for line in open(split_file).readlines()]
        
        self.calib_dir = os.path.join(data_dir, 'calib')
        self.image_dir = os.path.join(data_dir, 'image_2')
        self.velo_dir = os.path.join(data_dir, 'velodyne')
        if not self.is_test:
            self.label_dir = os.path.join(data_dir, 'label_2')

    def __len__(self):
        return len(self.sample_ids)

    def _parse_calib(self, filepath):
        """Reads calibration file and computes all necessary transformation matrices."""
        data = {}
        with open(filepath, 'r') as f:
            for line in f:
                key, value_str = line.split(':', 1)
                data[key] = np.array([float(v) for v in value_str.strip().split()])
        
        P0 = data['P0'].reshape(3, 4)
        
        tr_velo_to_cam = np.eye(4)
        tr_velo_to_cam[:3, :4] = data['Tr_velo_to_cam'].reshape(3, 4)
        
        R = tr_velo_to_cam[:3, :3]
        T = tr_velo_to_cam[:3, 3]
        tr_cam_to_velo = np.eye(4)
        tr_cam_to_velo[:3, :3] = R.T
        tr_cam_to_velo[:3, 3] = -R.T @ T

        return {'P0': P0, 'Tr_velo_to_cam': tr_velo_to_cam, 'Tr_cam_to_velo': tr_cam_to_velo}
    
    def _parse_labels(self, filepath):
        """Parses a label file into a list of dictionaries, without transformation."""
        labels = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15: continue
                
                labels.append({
                    'type': parts[0],
                    'bbox_2d': np.array([float(p) for p in parts[4:8]], dtype=np.float32),
                    'dims': np.array([float(p) for p in parts[8:11]], dtype=np.float32), # h, w, l
                    'loc_cam': np.array([float(p) for p in parts[11:14]], dtype=np.float32), # x, y, z in camera
                    'ry': float(parts[14])
                })
        return labels

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        calib = self._parse_calib(os.path.join(self.calib_dir, f'{sample_id}.txt'))
        tr_cam_to_velo = calib['Tr_cam_to_velo']

        # --- 1. Load Raw Point Cloud and Interpret into Camera Coordinates ---
        points_raw = np.fromfile(os.path.join(self.velo_dir, f'{sample_id}.bin'), dtype=np.float32).reshape(-1, 8)
        
        x_file, y_file, z_file = points_raw[:, 0], points_raw[:, 1], points_raw[:, 2]
        x_cam = -y_file
        y_cam = -z_file
        z_cam = x_file
        points_in_cam_coords = np.stack([x_cam, y_cam, z_cam], axis=1)
        
        # --- 2. Transform Points to Lidar Coordinate System ---
        points_cam_h = np.hstack((points_in_cam_coords, np.ones((points_in_cam_coords.shape[0], 1))))
        points_in_lidar_coords = (tr_cam_to_velo @ points_cam_h.T).T[:, :3]

        # --- 3. Assemble Final Point Features ---
        doppler = points_raw[:, 3]
        power = points_raw[:, 4]
        points_features = np.hstack((points_in_lidar_coords, doppler[:, np.newaxis], power[:, np.newaxis]))

        # --- 4. Load Image (Ready for future fusion tasks) ---
        image = cv2.imread(os.path.join(self.image_dir, f'{sample_id}.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        data_dict = {
            'points': points_features,
            'image': image,
            'frame_id': sample_id,
            'calib': calib
        }

        # --- 5. Load Labels and Transform to Lidar Coordinate System ---
        if not self.is_test:
            labels_cam = self._parse_labels(os.path.join(self.label_dir, f'{sample_id}.txt'))
            
            gt_boxes_lidar = []
            for label in labels_cam:
                loc_cam = label['loc_cam']
                dims = label['dims'] # h, w, l
                ry = label['ry']
                
                loc_cam_h = np.append(loc_cam, 1)
                loc_lidar = (tr_cam_to_velo @ loc_cam_h)[:3]
                
                heading = -(ry + np.pi / 2)
                
                gt_boxes_lidar.append(np.array([loc_lidar[0], loc_lidar[1], loc_lidar[2], dims[2], dims[1], dims[0], heading]))
            
            data_dict['gt_boxes'] = np.array(gt_boxes_lidar) if gt_boxes_lidar else np.empty((0, 7))
            data_dict['gt_names'] = np.array([l['type'] for l in labels_cam])
            
        return data_dict