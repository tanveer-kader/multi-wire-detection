import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from models.pointnet2_multi import PointNet2MultiDetector
from utils.data_loader_multi import create_dataloaders_multi
from utils.visualization import plot_training_history


class MultiWireLoss(nn.Module):
    """Combined loss for multiple wire detection."""
    
    def __init__(self, cls_weight=1.0, bbox_weight=5.0, rotation_weight=2.0):
        super(MultiWireLoss, self).__init__()
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.rotation_weight = rotation_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')
    
    def forward(self, cls_logits, bbox_preds, cls_targets, bbox_targets):
        """
        Args:
            cls_logits: [B, max_wires] classification logits
            bbox_preds: [B, max_wires, 7] predicted bbox parameters
            cls_targets: [B, max_wires] classification labels
            bbox_targets: [B, max_wires, 7] target bbox parameters
        """
        B, max_wires = cls_logits.shape
        
        # Classification loss for each wire
        cls_loss = self.bce_loss(cls_logits, cls_targets).mean()
        
        # Bounding box loss (only for wires that exist)
        mask = (cls_targets > 0.5).float().unsqueeze(-1)  # [B, max_wires, 1]
        
        # Center and size loss
        bbox_loss = (self.smooth_l1_loss(
            bbox_preds[:, :, :6], 
            bbox_targets[:, :, :6]
        ) * mask).sum() / (mask.sum() + 1e-8)
        
        # Rotation loss (separate because it's periodic)
        rotation_loss = (self.smooth_l1_loss(
            bbox_preds[:, :, 6:7],
            bbox_targets[:, :, 6:7]
        ) * mask).sum() / (mask.sum() + 1e-8)
        
        # Combined loss
        total_loss = (
            self.cls_weight * cls_loss + 
            self.bbox_weight * bbox_loss + 
            self.rotation_weight * rotation_loss
        )
        
        return total_loss, cls_loss, bbox_loss, rotation_loss


def calculate_accuracy(cls_logits, cls_targets, threshold=0.5):
    """Calculate classification accuracy for all wires."""
    predictions = torch.sigmoid(cls_logits) > threshold
    targets = cls_targets > 0.5
    correct = (predictions == targets).sum().item()
    total = cls_targets.numel()
    return correct / total


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_cls_loss = 0
    total_bbox_loss = 0
    total_rot_loss = 0
    total_accuracy = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        points = batch['points'].to(device)  # [B, N, 3]
        labels = batch['labels'].to(device)  # [B, max_wires]
        bboxes = batch['bboxes'].to(device)  # [B, max_wires, 7]
        
        # Forward pass
        optimizer.zero_grad()
        cls_logits, bbox_pred = model(points)
        
        # Calculate loss
        loss, cls_loss, bbox_loss, rot_loss = criterion(
            cls_logits, bbox_pred, labels, bboxes
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        accuracy = calculate_accuracy(cls_logits, labels)
        
        # Update metrics
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_bbox_loss += bbox_loss.item()
        total_rot_loss += rot_loss.item()
        total_accuracy += accuracy
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.4f}'
        })
    
    num_batches = len(train_loader)
    return {
        'loss': total_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches,
        'bbox_loss': total_bbox_loss / num_batches,
        'rot_loss': total_rot_loss / num_batches,
        'accuracy': total_accuracy / num_batches
    }


def validate_epoch(model, val_loader, criterion, device, epoch):
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0
    total_cls_loss = 0
    total_bbox_loss = 0
    total_rot_loss = 0
    total_accuracy = 0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            points = batch['points'].to(device)
            labels = batch['labels'].to(device)
            bboxes = batch['bboxes'].to(device)
            
            # Forward pass
            cls_logits, bbox_pred = model(points)
            
            # Calculate loss
            loss, cls_loss, bbox_loss, rot_loss = criterion(
                cls_logits, bbox_pred, labels, bboxes
            )
            
            # Calculate accuracy
            accuracy = calculate_accuracy(cls_logits, labels)
            
            # Update metrics
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_bbox_loss += bbox_loss.item()
            total_rot_loss += rot_loss.item()
            total_accuracy += accuracy
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.4f}'
            })
    
    num_batches = len(val_loader)
    return {
        'loss': total_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches,
        'bbox_loss': total_bbox_loss / num_batches,
        'rot_loss': total_rot_loss / num_batches,
        'accuracy': total_accuracy / num_batches
    }


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(f"cuda:{config['hardware']['gpu_id']}" 
                         if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders_multi(config, train_split=config['data']['train_split'])
    
    # Create model
    print("\nCreating multi-wire detection model...")
    max_wires = config['model'].get('max_wires', 6)
    model = PointNet2MultiDetector(
        num_classes=config['model']['num_classes'],
        input_channels=config['model']['input_channels'],
        max_wires=max_wires
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params} parameters")
    print(f"Detecting up to {max_wires} wires per point cloud")
    
    # Create loss function
    criterion = MultiWireLoss(
        cls_weight=config['loss']['classification_weight'],
        bbox_weight=config['loss']['bbox_weight'],
        rotation_weight=config['loss']['rotation_weight']
    )
    
    # Create optimizer
    if config['optimizer']['type'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['optimizer']['momentum'],
            weight_decay=config['training']['weight_decay']
        )
    
    # Learning rate scheduler
    if config['training']['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['lr_decay_step'],
            gamma=config['training']['lr_decay_rate']
        )
    elif config['training']['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=10
        )
    
    # Create checkpoint directory
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['logging']['result_dir'], exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(config['logging']['log_dir'])
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\n" + "="*50)
    print("Starting Multi-Wire Training")
    print("="*50)
    
    # Training loop
    for epoch in range(1, config['training']['num_epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config['training']['num_epochs']}")
        print(f"{'='*50}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        if epoch % config['validation']['eval_frequency'] == 0:
            val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
        else:
            val_metrics = {
                'loss': train_metrics['loss'], 
                'cls_loss': train_metrics['cls_loss'],
                'bbox_loss': train_metrics['bbox_loss'],
                'rot_loss': train_metrics['rot_loss'],
                'accuracy': train_metrics['accuracy']
            }
        
        # Update learning rate
        if config['training']['lr_scheduler'] == 'plateau':
            if epoch % config['validation']['eval_frequency'] == 0:
                scheduler.step(val_metrics['loss'])
        else:
            scheduler.step()
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Print metrics
        print(f"\nTrain Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if epoch % config['validation']['save_frequency'] == 0:
            checkpoint_path = os.path.join(
                config['logging']['checkpoint_dir'],
                f'checkpoint_multi_epoch_{epoch}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'max_wires': max_wires
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            best_model_path = os.path.join(
                config['logging']['checkpoint_dir'],
                'best_model_multi.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'max_wires': max_wires
            }, best_model_path)
            print(f" Saved best model with val_loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['validation']['early_stopping_patience']:
            print(f"\n Early stopping triggered after {epoch} epochs")
            break
    
    # Plot training history
    print("\n" + "="*50)
    print("Training Completed!")
    print("="*50)
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Total epochs: {epoch}")
    
    plot_training_history(history, save_dir=config['logging']['result_dir'])
    
    writer.close()
    print("\n Multi-wire training finished successfully!")
    print(f"Best model saved at: {os.path.join(config['logging']['checkpoint_dir'], 'best_model_multi.pth')}")
    print(f"Training history plot saved at: {os.path.join(config['logging']['result_dir'], 'training_history.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PointNet++ for Multi-Wire Detection')
    parser.add_argument('--config', type=str, default='config_multi.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)