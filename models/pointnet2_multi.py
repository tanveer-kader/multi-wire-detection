import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2 import PointNet2Backbone


class PointNet2MultiDetector(nn.Module):
    """
    Multi-wire detector: Predicts up to MAX_WIRES wires per point cloud.
    """
    def __init__(self, num_classes=1, input_channels=3, max_wires=6):
        super(PointNet2MultiDetector, self).__init__()
        
        self.max_wires = max_wires
        self.backbone = PointNet2Backbone(input_channels=input_channels)
        
        # Shared feature extraction
        self.shared_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification heads for each wire
        self.cls_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1)  # Wire present or not
            )
            for _ in range(max_wires)
        ])
        
        # Bounding box heads for each wire
        self.bbox_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 7)  # center(3) + size(3) + rotation(1)
            )
            for _ in range(max_wires)
        ])
    
    def forward(self, xyz, features=None):
        """
        Args:
            xyz: [B, N, 3]
            features: [B, N, C] or None
        
        Returns:
            cls_logits: [B, max_wires] - classification for each wire
            bbox_preds: [B, max_wires, 7] - bbox for each wire
        """
        B = xyz.shape[0]
        
        # Extract global features
        global_features = self.backbone(xyz, features)  # [B, 1024]
        shared_features = self.shared_fc(global_features)  # [B, 512]
        
        # Predict for each wire
        cls_logits_list = []
        bbox_preds_list = []
        
        for i in range(self.max_wires):
            cls_logit = self.cls_heads[i](shared_features)  # [B, 1]
            bbox_pred = self.bbox_heads[i](shared_features)  # [B, 7]
            
            cls_logits_list.append(cls_logit)
            bbox_preds_list.append(bbox_pred)
        
        cls_logits = torch.cat(cls_logits_list, dim=1)  # [B, max_wires]
        bbox_preds = torch.stack(bbox_preds_list, dim=1)  # [B, max_wires, 7]
        
        return cls_logits, bbox_preds


if __name__ == "__main__":
    # Test the model
    print("Testing PointNet2MultiDetector...")
    model = PointNet2MultiDetector(num_classes=1, input_channels=3, max_wires=6)
    xyz = torch.randn(2, 8192, 3)
    
    print(f"Input shape: {xyz.shape}")
    
    cls_logits, bbox_preds = model(xyz)
    
    print(f"Classification logits shape: {cls_logits.shape}")  # [2, 6]
    print(f"Bounding box predictions shape: {bbox_preds.shape}")  # [2, 6, 7]
    print("Multi-wire model test passed!")
    
    # Test individual predictions
    print(f"\nSample output:")
    print(f"Wire 1 classification logit: {cls_logits[0, 0].item():.4f}")
    print(f"Wire 1 bbox prediction: {bbox_preds[0, 0, :3].detach().numpy()}")