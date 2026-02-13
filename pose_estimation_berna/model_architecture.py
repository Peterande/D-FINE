import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from .dfine_library.src.core import YAMLConfig

class ASPP(nn.Module):
    """Simplified ASPP for better speed/accuracy balance"""
    
    def __init__(self, in_channels, out_channels, dilations=[1, 6, 12]):
        super().__init__()
        
        self.convs = nn.ModuleList()
        for dilation in dilations:
            if dilation == 1:
                conv = nn.Conv2d(in_channels, out_channels//len(dilations), 1, bias=False)
            else:
                conv = nn.Conv2d(in_channels, out_channels//len(dilations), 3, 
                               padding=dilation, dilation=dilation, bias=False)
            self.convs.append(nn.Sequential(
                conv, 
                nn.BatchNorm2d(out_channels//len(dilations)), 
                nn.ReLU(inplace=True)
            ))
        
        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels//len(dilations), 1, bias=False),
            nn.BatchNorm2d(out_channels//len(dilations)),
            nn.ReLU(inplace=True)
        )
        
        # Final projection
        total_channels = out_channels//len(dilations) * (len(dilations) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def forward(self, x):
        h, w = x.shape[-2:]
        features = []
        
        # Apply dilated convolutions
        for conv in self.convs:
            features.append(conv(x))
        
        # Global pooling branch
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=False)
        features.append(global_feat)
        
        # Combine and project
        combined = torch.cat(features, dim=1)
        return self.project(combined)


class FPN(nn.Module):
    """Improved FPN with better feature fusion"""
    
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1, bias=False) 
            for in_ch in in_channels_list
        ])
        
        # Output convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in in_channels_list
        ])
        
        # Feature fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(len(in_channels_list)))
        
    def forward(self, features):
        # Build laterals
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # Build top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], 
                mode='bilinear', align_corners=False
            )
            laterals[i-1] = laterals[i-1] + upsampled
        
        # Apply final convs
        outputs = [conv(lateral) for conv, lateral in zip(self.fpn_convs, laterals)]
        
        # Weighted feature fusion
        target_size = outputs[0].shape[-2:]
        fused_features = []
        
        for i, feat in enumerate(outputs):
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            fused_features.append(feat * self.fusion_weights[i])
        
        # Return highest resolution feature with weighted fusion
        return sum(fused_features)


class SegmentationHead(nn.Module):
    """Enhanced but stable segmentation head"""
    
    def __init__(self, in_channels_list, num_classes=7, feature_dim=256, dropout_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        
        logger.info(f"Creating enhanced segmentation head with channels: {in_channels_list}")
        
        # Improved FPN
        self.fpn = FPN(in_channels_list, feature_dim)
        
        # Simplified ASPP for multi-scale context
        self.aspp = ASPP(feature_dim, feature_dim)
        
        # Enhanced decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(feature_dim, feature_dim//2, 3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim//2),
            nn.ReLU(inplace=True),
            
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(feature_dim//2, num_classes, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, features):
        # Multi-scale feature fusion
        fused_feature = self.fpn(features)
        
        # Apply simplified ASPP
        context_feature = self.aspp(fused_feature)
        
        # Generate segmentation
        seg_logits = self.decoder(context_feature)
        
        return seg_logits


class DFineWithSegmentation(nn.Module):
    """D-FINE with enhanced segmentation"""
    
    def __init__(self, dfine_model, seg_head, freeze_detection=True):
        super().__init__()
        self.dfine_model = dfine_model
        self.seg_head = seg_head
        self.freeze_detection = freeze_detection
        
        if freeze_detection:
            self._freeze_detection()
    
    def _freeze_detection(self):
        frozen_params = 0
        total_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if 'seg_head' not in name:
                param.requires_grad = False
                frozen_params += param.numel()
        
        trainable_params = total_params - frozen_params
        
        logger.info(f"Frozen detection components:")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        logger.info(f"   Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    def forward(self, x, targets=None):
        # Get backbone features
        backbone_features = self.dfine_model.backbone(x)
        
        outputs = {}
        
        # Detection branch (frozen during training)
        if not self.training or not self.freeze_detection:
            with torch.no_grad() if self.freeze_detection else torch.enable_grad():
                det_outputs = self.dfine_model(x)
                outputs.update(det_outputs)
        
        # Segmentation branch
        seg_logits = self.seg_head(backbone_features)
        
        # Upsample to input resolution
        seg_logits = F.interpolate(
            seg_logits, size=x.shape[-2:], 
            mode='bilinear', align_corners=False
        )
        
        outputs['segmentation'] = seg_logits
        
        return outputs


class SegmentationModelWrapper(nn.Module):
    """Wrapper model for ONNX export with proper postprocessing"""
    
    def __init__(self, seg_model, postprocessor):
        super().__init__()
        self.seg_model = seg_model.eval()
        self.postprocessor = postprocessor
        
    def forward(self, images, orig_target_sizes):
        # Run the segmentation model
        outputs = self.seg_model(images)
        
        # Extract detection outputs for postprocessing
        det_outputs = {k: v for k, v in outputs.items() if k != 'segmentation'}
        
        # Process detection outputs
        processed_det = self.postprocessor(det_outputs, orig_target_sizes)
        
        # Get segmentation outputs
        seg_logits = outputs['segmentation']
        
        # Apply softmax to get probabilities
        seg_probs = torch.softmax(seg_logits, dim=1)
        
        # Get segmentation predictions
        seg_preds = torch.argmax(seg_logits, dim=1)
        
        return processed_det[0], processed_det[1], processed_det[2], seg_probs, seg_preds


def create_segmentation_model(config_path, original_weights_path, checkpoint_path, hyperparams, model_size="x"):
    """
    Create segmentation model with exact architecture from working exporter.
    
    Args:
        config_path: Path to D-FINE config
        original_weights_path: Path to original D-FINE weights  
        checkpoint_path: Path to trained segmentation checkpoint
        hyperparams: Hyperparameters extracted from checkpoint
        model_size: Model size (x, l, m, s, n)
        
    Returns:
        Complete segmentation model
    """
    logger.info("Creating segmentation model...")
    logger.info(f"Config: {config_path}")
    logger.info(f"Original weights: {original_weights_path}")
    logger.info(f"Model size: {model_size}")
    
    # Load original DFINE model
    logger.info("Loading DFINE configuration...")
    cfg = YAMLConfig(str(config_path))
    dfine_model = cfg.model
    
    # Load original weights
    logger.info("Loading original DFINE weights...")
    original_checkpoint = torch.load(original_weights_path, map_location='cpu')
    
    if 'ema' in original_checkpoint and 'module' in original_checkpoint['ema']:
        state_dict = original_checkpoint['ema']['module']
    elif 'model' in original_checkpoint:
        state_dict = original_checkpoint['model']
    else:
        state_dict = original_checkpoint
    
    dfine_model.load_state_dict(state_dict, strict=False)
    
    # Get backbone channels dynamically
    logger.info("Analyzing backbone architecture...")
    dfine_model.eval()
    dummy_input = torch.randn(1, 3, 640, 640)
    
    with torch.no_grad():
        backbone_features = dfine_model.backbone(dummy_input)
    
    backbone_channels = [feat.shape[1] for feat in backbone_features]
    logger.info(f"Detected backbone channels: {backbone_channels}")
    
    # Create segmentation head with EXACT architecture from working exporter
    logger.info("Creating segmentation head...")
    seg_head = SegmentationHead(
        in_channels_list=backbone_channels,
        num_classes=7,  # Pascal Person Parts
        feature_dim=hyperparams.get('feature_dim', 256),
        dropout_rate=hyperparams.get('dropout_rate', 0.1)
    )
    
    # Create combined model with EXACT architecture from working exporter
    logger.info("Creating combined DFINE segmentation model...")
    model = DFineWithSegmentation(
        dfine_model=dfine_model,
        seg_head=seg_head,
        freeze_detection=True  # Match working exporter
    )
    
    # Load trained weights
    logger.info(f"Loading trained segmentation weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict
    try:
        model.load_state_dict(state_dict, strict=True)
        logger.info(f"Loaded trained segmentation model successfully")
    except RuntimeError as e:
        logger.warning(f"Strict loading failed: {e}")
        # Try non-strict loading
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded with non-strict mode")
        if missing_keys:
            logger.info(f"   Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            logger.info(f"   Unexpected keys: {len(unexpected_keys)}")
    
    logger.info("Segmentation model creation completed")
    return model


def create_wrapper_model(seg_model, config_path=None):
    """Create wrapper model for ONNX export with proper postprocessing"""
    logger.info("Creating wrapper model for ONNX export...")
    postprocessor = None
    
    if config_path:
        try:
            cfg = YAMLConfig(str(config_path))
            postprocessor = cfg.postprocessor.deploy()
            logger.info("Loaded postprocessor from config")
        except Exception as e:
            logger.warning(f"Could not load postprocessor: {e}")
            logger.info("Using identity postprocessor")
            postprocessor = lambda x, y: (x.get('pred_logits', torch.empty(0)), 
                                         x.get('pred_boxes', torch.empty(0)), 
                                         torch.empty(0))
    else:
        logger.info("No config provided, using identity postprocessor")
        postprocessor = lambda x, y: (x.get('pred_logits', torch.empty(0)), 
                                     x.get('pred_boxes', torch.empty(0)), 
                                     torch.empty(0))
    
    wrapper = SegmentationModelWrapper(seg_model, postprocessor)
    logger.info("Wrapper model created successfully")
    return wrapper