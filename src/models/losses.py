# losses.py (Final Corrected Version for Multi-class)

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointPillarsLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, code_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.code_weights = code_weights if code_weights is not None else [1.0] * 7
        self.reg_loss_func = nn.SmoothL1Loss(reduction='none')

    def forward(self, cls_preds, box_preds, cls_labels, reg_labels):
        """
        Args:
            cls_preds (torch.Tensor): (B, num_anchors, num_classes)
            box_preds (torch.Tensor): (B, num_anchors, 7)
            cls_labels (torch.Tensor): (B, num_anchors), values 0 (bg), 1, 2, 3...
            reg_labels (torch.Tensor): (B, num_anchors, 7)
        """
        # --- 1. Get masks for positive and valid anchors ---
        pos_mask = (cls_labels > 0)
        
        # --- 2. Classification Loss (Focal Loss for Multi-class) ---
        num_classes = cls_preds.shape[-1]
        
        # Convert class labels to one-hot encoding
        # cls_labels has shape (B, num_anchors). We want (B, num_anchors, num_classes)
        # We handle the background class (0) by making the one-hot vector all zeros.
        one_hot_targets = F.one_hot(torch.clamp(cls_labels, min=0).long(), num_classes=num_classes + 1)
        # Remove the background class column (index 0) to match predictions
        one_hot_targets = one_hot_targets[:, :, 1:].float() # Shape: (B, num_anchors, num_classes)
        
        # Calculate focal loss
        cls_preds_sigmoid = torch.sigmoid(cls_preds)
        pt = torch.where(one_hot_targets == 1, cls_preds_sigmoid, 1 - cls_preds_sigmoid)
        
        alpha_factor = torch.where(one_hot_targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_factor * (1 - pt).pow(self.gamma)
        
        # Binary Cross Entropy loss, applied element-wise
        bce_loss = F.binary_cross_entropy_with_logits(cls_preds, one_hot_targets, reduction='none')
        
        # Only consider anchors that are not ignored (-1)
        valid_mask = (cls_labels >= 0).unsqueeze(-1) # Shape: (B, num_anchors, 1)
        cls_loss = (focal_weight * bce_loss * valid_mask).sum() / torch.clamp(pos_mask.sum(), min=1.0)

        # --- 3. Regression Loss (Smooth L1 Loss) ---
        # Only calculate regression loss for positive anchors
        pos_reg_mask = pos_mask.unsqueeze(-1).expand_as(reg_labels)
        
        # Apply weights to each component of the box regression
        weighted_code_weights = torch.tensor(self.code_weights, device=box_preds.device).view(1, 1, 7)

        reg_loss = self.reg_loss_func(box_preds, reg_labels)
        reg_loss = (reg_loss * weighted_code_weights * pos_reg_mask).sum() / torch.clamp(pos_mask.sum(), min=1.0)

        # --- 4. Total Loss ---
        total_loss = cls_loss + reg_loss
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'reg_loss': reg_loss
        }