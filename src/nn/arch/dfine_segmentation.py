"""
D-FINE with Body Parts Segmentation
"""

import torch
import torch.nn as nn
from ...core import register

__all__ = ["DFineSegmentation"]


@register()
class DFineSegmentation(nn.Module):
    """
    D-FINE with added body parts segmentation head
    """
    __inject__ = [
        "backbone",
        "neck", 
        "head",
        "seg_head"
    ]
    
    def __init__(
        self, 
        backbone: nn.Module,
        neck: nn.Module, 
        head: nn.Module,
        seg_head: nn.Module,
        freeze_detection=True
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.seg_head = seg_head
        self.freeze_detection = freeze_detection
        
        # Freeze detection components if needed
        if freeze_detection:
            self._freeze_detection()
    
    def _freeze_detection(self):
        """Freeze all detection-related parameters"""
        frozen_params = 0
        total_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if 'seg_head' not in name:
                param.requires_grad = False
                frozen_params += param.numel()
        
        trainable_params = total_params - frozen_params
        
        print(f"🔒 Frozen detection components:")
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   ❄️  Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"   🎯 Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    def forward(self, x, targets=None):
        # Get backbone features
        backbone_features = self.backbone(x)
        
        outputs = {}
        
        # Detection branch (frozen during training)
        if not self.training or not self.freeze_detection:
            neck_features = self.neck(backbone_features)
            det_outputs = self.head(neck_features)
            outputs.update(det_outputs)  # pred_logits, pred_boxes
        
        # Segmentation branch (trainable)
        seg_logits = self.seg_head(backbone_features)
        
        # Upsample to input resolution
        seg_logits = F.interpolate(
            seg_logits, size=x.shape[-2:], 
            mode='bilinear', align_corners=False
        )
        
        outputs['segmentation'] = seg_logits
        
        return outputs
    
    def deploy(self):
        """Prepare model for deployment"""
        self.eval()
        for m in self.modules():
            if m is not self and hasattr(m, 'deploy'):
                m.deploy()
        return self