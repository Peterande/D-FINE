#!/usr/bin/env python3
"""
Unified DFINE Segmentation Training Script
Single script supporting lightweight, standard, and advanced tiers
Uses tier-specific YAML configurations for all parameters
"""

import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from pathlib import Path

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
sys.path.append('src')

# Import core modules
from core.datasets import create_pascal_dataset
from core.losses import create_loss
from core.metrics import create_metrics_tracker
from core.models import load_pretrained_dfine, get_actual_backbone_channels, create_combined_model

# Import tier-specific models
from models import create_segmentation_head


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Unified DFINE Segmentation Training')
    
    # Required tier selection
    parser.add_argument('--tier', type=str, required=True,
                        choices=['lightweight', 'standard', 'advanced'],
                        help='Model tier to train (lightweight/standard/advanced)')
    
    # Optional overrides
    parser.add_argument('--config-dir', default='configs',
                        help='Directory containing tier configuration files')
    parser.add_argument('--dataset', default=None,
                        help='Override dataset path from config')
    parser.add_argument('--output-dir', default=None,
                        help='Override output directory from config')
    parser.add_argument('--resume', default=None,
                        help='Resume training from checkpoint')
    
    # System configuration
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Override number of data loading workers')
    
    # Logging
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--wandb-project', default=None,
                        help='Override wandb project name')
    parser.add_argument('--wandb-run-name', default=None,
                        help='Override wandb run name')
    
    # Debug options
    parser.add_argument('--dry-run', action='store_true',
                        help='Test configuration loading without training')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()


def parse_numeric_values(config):
    """Recursively parse string numeric values in config to proper types"""
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, str):
                # Try to parse scientific notation
                try:
                    if 'e-' in value.lower() or 'e+' in value.lower():
                        config[key] = float(value)
                    elif value.replace('.', '').replace('-', '').isdigit():
                        if '.' in value:
                            config[key] = float(value)
                        else:
                            config[key] = int(value)
                except (ValueError, AttributeError):
                    pass  # Keep as string if parsing fails
            elif isinstance(value, dict):
                parse_numeric_values(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        parse_numeric_values(item)
                    elif isinstance(item, str):
                        try:
                            if 'e-' in item.lower() or 'e+' in item.lower():
                                value[i] = float(item)
                            elif item.replace('.', '').replace('-', '').isdigit():
                                if '.' in item:
                                    value[i] = float(item)
                                else:
                                    value[i] = int(item)
                        except (ValueError, AttributeError):
                            pass


def load_tier_config(tier: str, config_dir: str) -> dict:
    """Load tier-specific configuration"""
    config_path = os.path.join(config_dir, f"{tier}.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    print(f"📋 Loading {tier} tier configuration from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load base config if specified
    if 'base' in config and config['base']:
        # Handle relative paths correctly
        base_filename = config['base']
        if not base_filename.startswith('/'):  # Relative path
            base_path = os.path.join(config_dir, base_filename)
        else:  # Absolute path
            base_path = base_filename
            
        print(f"📋 Loading base configuration from: {base_path}")
        
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Merge configs (tier config overrides base)
        merged_config = deep_merge_dict(base_config, config)
        
        # Parse numeric values
        parse_numeric_values(merged_config)
        
        return merged_config
    
    # Parse numeric values for non-base configs too
    parse_numeric_values(config)
    return config


def deep_merge_dict(base_dict: dict, override_dict: dict) -> dict:
    """Deep merge two dictionaries"""
    result = base_dict.copy()
    
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    
    return result


def setup_device(device_arg: str) -> torch.device:
    """Setup compute device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"🚀 Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def create_model(config: dict, tier: str, device: torch.device):
    """Create the segmentation model based on tier"""
    print(f"🏗️  Creating {tier} tier segmentation model...")
    
    # Load pretrained DFINE model
    dfine_config = config['dfine']['config_path']
    dfine_checkpoint = config['dfine']['checkpoint_path']
    
    dfine_model = load_pretrained_dfine(dfine_config, dfine_checkpoint)
    
    # Get backbone channels
    backbone_channels = get_actual_backbone_channels(dfine_model)
    print(f"   Detected backbone channels: {backbone_channels}")
    
    # Get model-specific parameters
    model_config = config.get('model', {})
    seg_head_config = model_config.get('segmentation_head', {})
    
    # Create tier-specific segmentation head
    seg_head = create_segmentation_head(
        tier=tier,
        in_channels_list=backbone_channels,
        num_classes=config['dataset']['num_classes'],
        feature_dim=seg_head_config.get('feature_dim', 256),
        dropout_rate=seg_head_config.get('dropout_rate', 0.1)
    )
    
    # Create combined model
    model = create_combined_model(
        dfine_model=dfine_model,
        seg_head=seg_head,
        freeze_detection=config['training'].get('freeze_detection', True)
    )
    
    model = model.to(device)
    
    # Print model info
    model_info = model.get_model_info()
    print(f"📊 Model Information:")
    print(f"   Tier: {model_info['tier']}")
    print(f"   Total parameters: {model_info['total_params']:,}")
    print(f"   Trainable parameters: {model_info['trainable_params']:,}")
    print(f"   Model size: {model_info['model_size_mb']:.2f} MB")
    
    return model


def create_datasets(config: dict, tier: str, dataset_override: str = None):
    """Create training and validation datasets"""
    print("📊 Creating datasets...")
    
    dataset_config = config['dataset']
    training_config = config['training']
    
    # Use override or config dataset path
    dataset_root = dataset_override or dataset_config['root_dir']
    
    # Training dataset with tier-specific augmentations
    train_dataset = create_pascal_dataset(
        root_dir=dataset_root,
        split='train',
        image_size=dataset_config['image_size'],
        tier=tier,
        multi_scale=training_config.get('multi_scale_training', False)
    )
    
    # Validation dataset without augmentations
    val_dataset = create_pascal_dataset(
        root_dir=dataset_root,
        split='val',
        image_size=dataset_config['image_size'],
        tier=tier,
        multi_scale=False
    )
    
    # Create data loaders
    system_config = config.get('system', {})
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=system_config.get('num_workers', 4),
        pin_memory=system_config.get('pin_memory', True),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=system_config.get('num_workers', 4),
        pin_memory=system_config.get('pin_memory', True)
    )
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


def create_optimizer_and_scheduler(model, config: dict):
    """Create optimizer and learning rate scheduler"""
    training_config = config['training']
    
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Create optimizer
    optimizer = optim.AdamW(
        trainable_params,
        lr=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 1e-4)
    )
    
    # Create scheduler
    scheduler_config = training_config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'cosine_warmup')
    
    if scheduler_type == 'cosine_warmup':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config.get('T_0', 15),
            T_mult=scheduler_config.get('T_mult', 2),
            eta_min=scheduler_config.get('eta_min', 1e-6)
        )
    else:
        # Default to cosine scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=15, T_mult=2, eta_min=1e-6
        )
    
    print(f"🎯 Optimizer: AdamW with {len(trainable_params)} parameter groups")
    print(f"📈 Scheduler: {scheduler_type}")
    
    return optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config, args):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_miou = 0.0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        seg_pred = outputs['segmentation']
        
        # Calculate loss (handle different loss function signatures)
        if hasattr(criterion, 'forward') and 'boundary' in str(criterion.forward.__code__.co_varnames):
            # Advanced loss with boundary prediction
            boundary_pred = outputs.get('boundary', None)
            loss, loss_dict = criterion(seg_pred, masks, boundary_pred)
        else:
            # Standard loss
            loss, loss_dict = criterion(seg_pred, masks)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        gradient_clipping = config['training'].get('gradient_clipping', 1.0)
        if gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)
        
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            from core.metrics import compute_miou
            miou = compute_miou(seg_pred, masks)
        
        # Update running averages
        total_loss += loss.item()
        total_miou += miou
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'mIoU': f'{miou:.3f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        
        # Log batch metrics
        if batch_idx % 20 == 0 and not args.no_wandb:
            log_dict = {
                'train/batch_loss': loss.item(),
                'train/batch_miou': miou,
                'train/lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch
            }
            
            # Add loss components if available
            if isinstance(loss_dict, dict):
                for key, value in loss_dict.items():
                    if key != 'total':
                        log_dict[f'train/batch_{key}'] = value
            
            wandb.log(log_dict)
    
    avg_loss = total_loss / num_batches
    avg_miou = total_miou / num_batches
    
    return avg_loss, avg_miou


def validate_epoch(model, val_loader, criterion, device, config):
    """Validate for one epoch"""
    model.eval()
    
    total_loss = 0.0
    total_miou = 0.0
    num_batches = len(val_loader)
    
    # Create metrics tracker for detailed evaluation
    dataset_config = config['dataset']
    metrics_tracker = create_metrics_tracker(
        num_classes=dataset_config['num_classes'],
        class_names=dataset_config.get('class_names', None)
    )
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            seg_pred = outputs['segmentation']
            
            # Calculate loss
            if hasattr(criterion, 'forward') and 'boundary' in str(criterion.forward.__code__.co_varnames):
                boundary_pred = outputs.get('boundary', None)
                loss, loss_dict = criterion(seg_pred, masks, boundary_pred)
            else:
                loss, loss_dict = criterion(seg_pred, masks)
            
            # Calculate metrics
            from core.metrics import compute_miou
            miou = compute_miou(seg_pred, masks)
            
            # Update metrics tracker
            metrics_tracker.update(seg_pred, masks)
            
            # Update running averages
            total_loss += loss.item()
            total_miou += miou
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'mIoU': f'{miou:.3f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_miou = total_miou / num_batches
    
    # Get detailed metrics
    detailed_metrics = metrics_tracker.get_summary()
    
    return avg_loss, avg_miou, detailed_metrics


def save_checkpoint(model, optimizer, scheduler, epoch, best_miou, config, args, filename=None):
    """Save training checkpoint"""
    output_config = config['output']
    output_dir = args.output_dir or output_config['base_dir']
    
    if filename is None:
        filename = f'checkpoint_epoch_{epoch}.pth'
    
    filepath = os.path.join(output_dir, filename)
    
    # Save hyperparameters from config
    hyperparameters = {
        'tier': args.tier,
        'feature_dim': config.get('model', {}).get('segmentation_head', {}).get('feature_dim', 256),
        'dropout_rate': config.get('model', {}).get('segmentation_head', {}).get('dropout_rate', 0.1),
        'num_classes': config['dataset']['num_classes'],
        'image_size': config['dataset']['image_size']
    }
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_miou': best_miou,
        'config': config,
        'args': vars(args),
        'hyperparameters': hyperparameters,
        'model_info': model.get_model_info()
    }, filepath)
    
    print(f"💾 Saved checkpoint: {filepath}")


def setup_wandb(config: dict, args):
    """Setup Weights & Biases logging"""
    if args.no_wandb:
        return
    
    logging_config = config.get('logging', {})
    wandb_config = logging_config.get('wandb', {})
    
    # Determine project name
    project_name = (
        args.wandb_project or 
        wandb_config.get('project') or 
        f'dfine-{args.tier}-segmentation'
    )
    
    # Determine run name
    run_name = (
        args.wandb_run_name or 
        f'{args.tier}_seg_{config["training"]["epochs"]}ep'
    )
    
    # Flatten config for wandb
    wandb_config_dict = {}
    
    def flatten_dict(d, parent_key='', sep='/'):
        if d is None:  # Handle None values
            return {}
        
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if v is None:
                items.append((new_key, None))
            elif isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    wandb_config_dict.update(flatten_dict(config))
    wandb_config_dict.update(vars(args))
    
    wandb.init(
        project=project_name,
        name=run_name,
        config=wandb_config_dict,
        tags=wandb_config.get('tags', []) + [args.tier]
    )
    
    print(f"📊 Initialized wandb: project={project_name}, run={run_name}")


def handle_backbone_unfreezing(model, optimizer, epoch, config):
    """Handle backbone unfreezing based on configuration"""
    training_config = config['training']
    
    if not training_config.get('freeze_detection', True):
        return  # Detection is not frozen
    
    unfreeze_epoch = training_config.get('unfreeze_epoch', None)
    if unfreeze_epoch and epoch == unfreeze_epoch:
        print("🔓 Unfreezing detection backbone")
        model.unfreeze_detection()
        
        # Add backbone parameters to optimizer with lower learning rate
        backbone_params = [p for p in model.dfine_model.backbone.parameters() if p.requires_grad]
        if backbone_params:
            backbone_lr_factor = training_config.get('backbone_lr_factor', 0.1)
            optimizer.add_param_group({
                'params': backbone_params,
                'lr': training_config['learning_rate'] * backbone_lr_factor
            })
            print(f"   Added {len(backbone_params)} backbone parameters with LR factor {backbone_lr_factor}")


def main():
    """Main training function"""
    args = parse_args()
    
    # Load tier configuration
    try:
        config = load_tier_config(args.tier, args.config_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"Available configs in {args.config_dir}: {os.listdir(args.config_dir)}")
        sys.exit(1)
    
    # Override config with command line arguments
    if args.dataset:
        config['dataset']['root_dir'] = args.dataset
    if args.num_workers:
        config.setdefault('system', {})['num_workers'] = args.num_workers
    
    # Print configuration summary
    print("=" * 80)
    print(f"🚀 UNIFIED DFINE SEGMENTATION TRAINING - {args.tier.upper()} TIER")
    print("=" * 80)
    print(f"📋 Configuration:")
    print(f"   Tier: {args.tier}")
    print(f"   Dataset: {config['dataset']['root_dir']}")
    print(f"   Image size: {config['dataset']['image_size']}")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Epochs: {config['training']['epochs']}")
    print(f"   Learning rate: {config['training']['learning_rate']}")
    print(f"   Multi-scale: {config['training'].get('multi_scale_training', False)}")
    
    # Print tier-specific features
    if args.tier == 'lightweight':
        print(f"🚀 Lightweight features: Speed optimized, mobile-ready")
    elif args.tier == 'standard':
        print(f"⚖️  Standard features: Balanced performance and accuracy")
    elif args.tier == 'advanced':
        print(f"🎯 Advanced features: Maximum accuracy, attention mechanisms")
    
    print("=" * 80)
    
    if args.dry_run:
        print("🧪 Dry run completed - configuration loaded successfully")
        return
    
    # Setup device
    device = setup_device(args.device)
    
    # Create output directory
    output_config = config['output']
    output_dir = args.output_dir or output_config['base_dir']
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    
    # Initialize wandb
    setup_wandb(config, args)
    
    # Create model
    model = create_model(config, args.tier, device)
    
    # Create datasets
    train_loader, val_loader = create_datasets(config, args.tier, args.dataset)
    
    # Create loss function
    loss_config = config.get('loss', {})
    loss_type = loss_config.get('type', args.tier)
    
    criterion = create_loss(
        tier=loss_type,
        num_classes=config['dataset']['num_classes'],
        ignore_index=loss_config.get('ignore_index', 255)
    )
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    # Resume training if specified
    start_epoch = 0
    best_miou = 0.0
    
    if args.resume:
        print(f"📚 Resuming training from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_miou = checkpoint.get('best_miou', 0.0)
        print(f"   Resumed from epoch {start_epoch}, best mIoU: {best_miou:.4f}")
    
    # Training loop
    epochs = config['training']['epochs']
    save_every = output_config.get('save_every', 10)
    
    print(f"\n🚀 Starting training for {epochs - start_epoch} epochs...")
    print(f"📁 Output directory: {output_dir}")
    
    for epoch in range(start_epoch, epochs):
        print(f"\n📅 Epoch {epoch+1}/{epochs}")
        
        # Handle backbone unfreezing
        handle_backbone_unfreezing(model, optimizer, epoch, config)
        
        # Train
        train_loss, train_miou = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config, args
        )
        
        # Validate
        val_loss, val_miou, detailed_metrics = validate_epoch(
            model, val_loader, criterion, device, config
        )
        
        # Update scheduler
        scheduler.step()
        
        # Print epoch results
        print(f"📊 Train - Loss: {train_loss:.4f}, mIoU: {train_miou:.3f}")
        print(f"📊 Val   - Loss: {val_loss:.4f}, mIoU: {val_miou:.3f}")
        
        # Print tier-specific performance info
        targets = config.get('targets', {})
        if 'accuracy' in targets:
            target_miou = targets['accuracy'].get('miou', 0.0)
            if target_miou > 0:
                progress = (val_miou / target_miou) * 100
                print(f"🎯 Target progress: {progress:.1f}% of {target_miou:.3f} mIoU target")
        
        # Log to wandb
        if not args.no_wandb:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/miou': train_miou,
                'val/loss': val_loss,
                'val/miou': val_miou,
                'lr': optimizer.param_groups[0]['lr']
            }
            
            # Add detailed metrics
            for key, value in detailed_metrics.items():
                log_dict[f'val/{key}'] = value
            
            wandb.log(log_dict)
        
        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            save_checkpoint(model, optimizer, scheduler, epoch, best_miou, config, args, 'best_model.pth')
            print(f"🏆 New best model! mIoU: {best_miou:.4f}")
        
        # Save regular checkpoint
        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_miou, config, args)
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best validation mIoU: {best_miou:.4f}")
    print(f"📁 Models saved in: {output_dir}")
    
    # Print tier-specific final summary
    targets = config.get('targets', {})
    if 'accuracy' in targets:
        target_miou = targets['accuracy'].get('miou', 0.0)
        if target_miou > 0:
            if best_miou >= target_miou:
                print(f"✅ Target achieved! {best_miou:.4f} >= {target_miou:.4f}")
            else:
                print(f"📊 Target progress: {(best_miou/target_miou)*100:.1f}% of {target_miou:.4f} target")
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()