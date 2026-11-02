# pointpillars_model.py (Final Version)

import torch
import torch.nn as nn
import torch.nn.functional as F

class PillarFeatureNet(nn.Module):
    """
    Pillar Feature Net (PFN).
    Encodes points within each pillar into a feature vector.
    """
    def __init__(self, num_input_features, num_filters, out_channels):
        super().__init__()
        self.num_input_features = num_input_features
        
        # PointNet-like structure: Convolution -> BatchNorm -> ReLU
        self.conv1 = nn.Conv1d(num_input_features, num_filters[0], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(num_filters[0])

    def forward(self, pillar_features, pillar_indices):
        """
        Args:
            pillar_features (torch.Tensor): (D, N, C) -> (num_pillars, max_points, num_features)
        Returns:
            torch.Tensor: Pillar-wise features (D, out_channels).
        """
        # Permute for Conv1d: (D, N, C) -> (D, C, N)
        x = pillar_features.permute(0, 2, 1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Max pooling over the points dimension (N)
        x_max = torch.max(x, dim=2, keepdim=False)[0]
        
        return x_max


class PointPillarScatter(nn.Module):
    """
    Scatters pillar features into a 2D BEV map.
    """
    def __init__(self, output_shape, num_bev_features):
        super().__init__()
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        self.num_bev_features = num_bev_features

    def forward(self, pillar_features, pillar_coords):
        """
        Args:
            pillar_features (torch.Tensor): (num_pillars, C)
            pillar_coords (torch.Tensor): (num_pillars, 4) with (batch_idx, z, y, x)
        Returns:
            torch.Tensor: BEV feature map (batch_size, C, H, W).
        """
        batch_size = pillar_coords[:, 0].max().int().item() + 1
        
        bev_feature_map = torch.zeros(
            batch_size, self.num_bev_features, self.ny, self.nx,
            dtype=pillar_features.dtype, device=pillar_features.device
        )
        
        batch_indices = pillar_coords[:, 0].long()
        y_indices = pillar_coords[:, 2].long()
        x_indices = pillar_coords[:, 3].long()
        
        bev_feature_map[batch_indices, :, y_indices, x_indices] = pillar_features
        
        return bev_feature_map


class Backbone2D(nn.Module):
    """
    2D CNN backbone for processing the BEV feature map.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # A simplified backbone with two downsampling blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class DetectionHead(nn.Module):
    """
    Predicts bounding boxes from the final BEV feature map.
    """
    def __init__(self, in_channels, num_classes, num_anchors):
        super().__init__()
        
        self.conv_cls = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1)
        self.conv_box = nn.Conv2d(in_channels, num_anchors * 7, kernel_size=1) # 7 box parameters (x,y,z,l,w,h,ry)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Feature map from the backbone (B, C, H, W).
        Returns:
            dict: Dictionary with raw convolution outputs.
        """
        cls_preds = self.conv_cls(x)
        box_preds = self.conv_box(x)
        
        return {
            'cls_preds': cls_preds,
            'box_preds': box_preds
        }


class PointPillars(nn.Module):
    """
    Main PointPillars model connecting all sub-modules.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.pfn = PillarFeatureNet(**config['pfn'])
        self.scatter = PointPillarScatter(**config['scatter'])
        self.backbone = Backbone2D(**config['backbone'])
        self.head = DetectionHead(**config['head'])

    def forward(self, pillar_features, pillar_coords, pillar_indices=None):
        """
        Args:
            pillar_features: Output from voxelization.
            pillar_coords: Output from voxelization.
        Returns:
            dict: Final predictions from the detection head.
        """
        encoded_pillars = self.pfn(pillar_features, pillar_indices)
        bev_feature_map = self.scatter(encoded_pillars, pillar_coords)
        bev_features = self.backbone(bev_feature_map)
        predictions = self.head(bev_features)
        
        return predictions