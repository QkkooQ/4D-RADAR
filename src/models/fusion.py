# Fusion model adapted from the original competition prototype.

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import our complete and corrected fusion modules
from .fusion_modules import ImageEncoder, BEVFeatureLifter

# ======================================================================================
# --- Part 1: Core Pillar-based Components (UNCHANGED) ---
# These are the fundamental building blocks from our original pure-lidar model.
# ======================================================================================

class PillarFeatureNet(nn.Module):
    """
    Pillar Feature Net (PFN).
    Encodes points within each pillar into a single feature vector.
    """
    def __init__(self, num_input_features=5, num_filters=(64,), out_channels=64, **kwargs):
        super().__init__()
        self.num_input_features = num_input_features
        
        # PointNet-like structure: Convolution -> BatchNorm -> ReLU
        self.conv1 = nn.Conv1d(num_input_features, num_filters[0], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(num_filters[0])

    def forward(self, pillar_features, pillar_indices=None):
        """
        Args:
            pillar_features (torch.Tensor): (D, N, C) -> (num_pillars, max_points, num_features)
        Returns:
            torch.Tensor: Pillar-wise features (D, out_channels).
        """
        # Permute for Conv1d: (D, N, C) -> (D, C, N)
        x = pillar_features.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        
        # Max pooling over the points dimension (N)
        x_max = torch.max(x, dim=2, keepdim=False)[0]
        
        return x_max

# ==============================================================================
# --- 请用下面的代码完整替换你原来的 PointPillarScatter 类 ---
# ==============================================================================
class PointPillarScatter(nn.Module):
    """
    Scatters pillar features into a 2D BEV map.
    
    **FINAL CORRECTION**: This version computes its output shape internally from
    point_cloud_range and voxel_size to GUARANTEE alignment with BEVFeatureLifter.
    It no longer relies on the ambiguous `output_shape` from the config.
    """
    def __init__(self, point_cloud_range, voxel_size, num_bev_features=64, **kwargs):
        super().__init__()
        self.num_bev_features = num_bev_features

        # Compute the BEV map dimensions internally and definitively.
        # H corresponds to X-axis, W corresponds to Y-axis.
        self.bev_height = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.bev_width = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

    def forward(self, pillar_features, pillar_coords):
        batch_size = pillar_coords[:, 0].max().int().item() + 1
        
        bev_feature_map = torch.zeros(
            batch_size,
            self.num_bev_features,
            self.bev_height,
            self.bev_width,
            dtype=pillar_features.dtype,
            device=pillar_features.device
        )
        
        batch_indices = pillar_coords[:, 0].long()
        y_voxel_indices = pillar_coords[:, 2].long()
        x_voxel_indices = pillar_coords[:, 3].long()
        
        x_bev_indices = torch.clamp(x_voxel_indices, 0, self.bev_height - 1)
        y_bev_indices = torch.clamp(y_voxel_indices, 0, self.bev_width - 1)
        
        bev_feature_map[batch_indices, :, x_bev_indices, y_bev_indices] = pillar_features
        
        return bev_feature_map
class Backbone2D(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, **kwargs):
        super().__init__()
        # --- CRITICAL FIX: Replace BatchNorm with GroupNorm ---
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 64), # num_groups=32, num_channels=64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 128), # num_groups=32, num_channels=128
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
        )
        # ----------------------------------------------------
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

class DetectionHead(nn.Module):
    """
    Predicts bounding boxes from the final BEV feature map.
    """
    def __init__(self, in_channels=128, num_classes=3, num_anchors=6, **kwargs):
        super().__init__()
        
        self.conv_cls = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1)
        self.conv_box = nn.Conv2d(in_channels, num_anchors * 7, kernel_size=1) # 7 box parameters

    def forward(self, x):
        """ Returns raw convolution outputs. """
        cls_preds = self.conv_cls(x)
        box_preds = self.conv_box(x)
        
        return {'cls_preds': cls_preds, 'box_preds': box_preds}

# ======================================================================================
# --- Part 2: The New Fusion Model ---
# This class orchestrates the lidar and image branches and fuses their features.
# ======================================================================================

class PointPillarsFusion(nn.Module):
    """
    Main PointPillars model with an integrated image fusion branch.
    This is the final, corrected version that accepts a config dictionary.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # --- Lidar Branch ---
        self.pfn = PillarFeatureNet(**config['pfn'])
        
        # --- 关键修正：为 PointPillarScatter 传递正确的参数 ---
        scatter_config = {
            'point_cloud_range': config['point_cloud_range'],
            'voxel_size': config['voxel_size'],
            'num_bev_features': config['pfn']['out_channels']
        }
        self.scatter = PointPillarScatter(**scatter_config)
        
        # --- Image Branch ---
        self.image_encoder = ImageEncoder(**config['image_encoder'])
        
        # Prepare configurations for the BEVFeatureLifter
        grid_conf = {
            'xbound': [config['point_cloud_range'][0], config['point_cloud_range'][3], config['voxel_size'][0]],
            'ybound': [config['point_cloud_range'][1], config['point_cloud_range'][4], config['voxel_size'][1]],
        }
        cam_conf = {
            'W': 640, 
            'H': 480,
        }
        self.image_lifter = BEVFeatureLifter(
            grid_config=grid_conf,
            cam_config=cam_conf,
            out_channels=config['image_encoder']['out_channels']
        )
        
        # --- Fusion and Detection ---
        # Calculate the number of input channels for the backbone after fusion
        fusion_in_channels = config['scatter']['num_bev_features'] + config['image_encoder']['out_channels']
        
        # Create backbone config, overriding the in_channels
        backbone_config = {**config['backbone'], 'in_channels': fusion_in_channels}
        self.backbone = Backbone2D(**backbone_config)
        
        # Create head config
        self.head = DetectionHead(**config['head'])

    def forward(self, batch, debug=False):
        """
        Model's forward pass.
        
        Args:
            batch (dict): A dictionary containing the input tensors.
            debug (bool): If True, returns intermediate feature maps for visualization.
        
        Returns:
            dict or tuple: Model predictions, and optionally intermediate features.
        """
        pillar_features = batch['pillar_features']
        pillar_coords = batch['pillar_coords']
        
        # --- Lidar Branch ---
        encoded_pillars = self.pfn(pillar_features)
        lidar_bev_features = self.scatter(encoded_pillars, pillar_coords)
        
        # --- Image Branch ---
        image = batch['image']
        calib = batch['calib']
        image_features = self.image_encoder(image)
        image_bev_features = self.image_lifter(image_features, calib)
        
        # --- Fusion & Detection ---
        fused_bev_features = torch.cat([lidar_bev_features, image_bev_features], dim=1)
        
        bev_features = self.backbone(fused_bev_features)
        predictions = self.head(bev_features)
        
        if debug:
            # In debug mode, return intermediate features for alignment visualization
            return predictions, lidar_bev_features, image_bev_features
        
        return predictions
    

# ==============================================================================
# --- 在文件末尾添加这个新的纯雷达模型类 ---
# ==============================================================================
class PointPillarsLidarOnly(nn.Module):
    """
    A simplified, LiDAR-only version of the PointPillars model.
    This is used to establish a baseline and debug the LiDAR branch independently.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Pillar Feature Net
        self.pfn = PillarFeatureNet(**config['pfn'])
        
        # 2. Point Pillar Scatter
        # 使用我们最终修正过的、能自计算尺寸的 Scatter
        scatter_config = {
            'point_cloud_range': config['point_cloud_range'],
            'voxel_size': config['voxel_size'],
            'num_bev_features': config['pfn']['out_channels']
        }
        self.scatter = PointPillarScatter(**scatter_config)
        
        # 3. 2D Backbone
        # 输入通道数现在就是 scatter 的输出通道数，不再需要融合
        backbone_config = {
            **config['backbone'], 
            'in_channels': config['pfn']['out_channels']
        }
        self.backbone = Backbone2D(**backbone_config)
        
        # 4. Detection Head
        self.head = DetectionHead(**config['head'])

    def forward(self, batch):
        """
        Forward pass for the LiDAR-only model.
        """
        pillar_features = batch['pillar_features']
        pillar_coords = batch['pillar_coords']
        
        # Pass data through the LiDAR pipeline
        encoded_pillars = self.pfn(pillar_features)
        lidar_bev_features = self.scatter(encoded_pillars, pillar_coords)
        bev_features = self.backbone(lidar_bev_features)
        predictions = self.head(bev_features)
        
        return predictions
# ==============================================================================
# --- 添加结束 ---
# ============================================================================
