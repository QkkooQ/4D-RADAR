# fusion_modules.py (Final Corrected Version)

import torch
import torch.nn as nn
import torch.nn.functional as F  # <-- CRITICAL FIX: Import the functional module
import torchvision.models as models
import numpy as np

class ImageEncoder(nn.Module):
    """
    Encodes the input image to extract rich semantic features.
    Uses a pre-trained ResNet-18 backbone.
    """
    def __init__(self, out_channels=64):
        super().__init__()
        # Use a pre-trained ResNet-18, loading the recommended default weights
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # We take the output of the early layers (before the final blocks)
        # This gives us a feature map with a reasonable spatial resolution
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-3])
        
        # Add a final 1x1 convolution to adjust the number of channels to our desired output
        # ResNet-18's layer2 output has 128 channels
        self.final_conv = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x):
        # x is expected to be a batch of images, shape (B, 3, H, W)
        x = self.feature_extractor(x)
        x = self.final_conv(x)
        return x


class BEVFeatureLifter(nn.Module):
    """
    Lifts 2D image features to 3D space and splats them onto a BEV grid.
    This is a simplified implementation of the LSS (Lift, Splat, Shoot) concept,
    designed to be a runnable baseline.
    """
    def __init__(self, grid_config, cam_config, out_channels):
        super().__init__()
        self.grid_config = grid_config
        self.cam_config = cam_config
        self.out_channels = out_channels

        # Create a set of discrete depth bins. (Start, Stop, Num points)
        self.depth_bins = torch.arange(
            start=1.0, 
            end=50.0, 
            step=(50.0 - 1.0) / 40.0, 
            dtype=torch.float32
        )
        self.num_depth_bins = len(self.depth_bins)
        
        # The depth network predicts a depth distribution for each pixel.
        # Input channels are from the image encoder.
        # Output channels are (context features + depth distribution logits).
        self.depth_net = nn.Conv2d(out_channels, out_channels + self.num_depth_bins, kernel_size=1)

    def forward(self, image_features, calib):
        """
        Args:
            image_features (torch.Tensor): Features from the image encoder (B, C, H_feat, W_feat).
            calib (dict): Calibration dictionary (unused in this simplified version, but required for the interface).
        
        Returns:
            torch.Tensor: BEV features from the image (B, C, H_bev, W_bev).
        """
        B, C, H_feat, W_feat = image_features.shape
        
        # 1. Predict depth distribution and context features for each pixel
        combined_features = self.depth_net(image_features)
        context_features = combined_features[:, :self.out_channels, ...]
        depth_logits = combined_features[:, self.out_channels:, ...]
        
        # Apply softmax over the depth dimension to get a probability distribution
        depth_dist = depth_logits.softmax(dim=1) # Shape: (B, D, H_feat, W_feat)
        
        # 2. Lift: Create a "feature cloud" by weighting context features by depth probabilities
        # Reshape for broadcasting:
        # context_features: (B, C, H, W) -> (B, C, 1, H, W)
        # depth_dist:       (B, D, H, W) -> (B, 1, D, H, W)
        context_features = context_features.unsqueeze(2)
        depth_dist = depth_dist.unsqueeze(1)
        
        # Element-wise multiplication creates the lifted features
        # Shape: (B, C, D, H_feat, W_feat)
        lifted_features = depth_dist * context_features
        
        # 3. Splat (Simplified for Baseline): Project features onto the BEV grid
        # A full LSS implementation would perform a complex geometric transformation here.
        # For our runnable baseline, we will use a simplified approach to get a BEV map
        # of the correct shape, allowing the end-to-end training to proceed.
        
        # First, we average the features across the depth dimension. This collapses the
        # feature cloud into a 2D feature map, conceptually similar to an ortho-projection.
        # (B, C, D, H_feat, W_feat) -> (B, C, H_feat, W_feat)
        pooled_features = torch.mean(lifted_features, dim=2)
        
        # Then, we use adaptive average pooling to resize this feature map to match the
        # desired BEV grid dimensions. This simulates the "splatting" process.
        bev_h = self.grid_config['x_res']
        bev_w = self.grid_config['y_res']
        
        # (B, C, H_feat, W_feat) -> (B, C, bev_h, bev_w)
        image_bev_features = F.adaptive_avg_pool2d(pooled_features, (bev_h, bev_w))

        return image_bev_features