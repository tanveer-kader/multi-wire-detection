import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.preprocessing import (
    load_ply_file, load_label_file, normalize_point_cloud,
    sample_points, rotate_point_cloud, jitter_point_cloud,
    random_flip_point_cloud, random_scale_point_cloud
)


class WireDetectionMultiDataset(Dataset):
    def __init__(self, ply_dir, label_dir, num_points=8192, augmentation=True, split='train', max_wires=6):
        """
        Multi-Wire Detection Dataset
        
        Args:
            ply_dir: Directory containing .ply files
            label_dir: Directory containing .txt label files
            num_points: Number of points to sample
            augmentation: Whether to apply data augmentation
            split: 'train' or 'val'
            max_wires: Maximum number of wires to detect
        """
        self.ply_dir = ply_dir
        self.label_dir = label_dir
        self.num_points = num_points
        self.augmentation = augmentation
        self.split = split
        self.max_wires = max_wires
        
        # Get list of files
        self.ply_files = sorted([f for f in os.listdir(ply_dir) if f.endswith('.ply')])
        
        print(f"Found {len(self.ply_files)} PLY files in {ply_dir}")
        
        # Verify corresponding label files exist
        self.valid_samples = []
        for ply_file in self.ply_files:
            base_name = os.path.splitext(ply_file)[0]
            label_file = base_name + '.txt'
            label_path = os.path.join(label_dir, label_file)
            
            if os.path.exists(label_path):
                self.valid_samples.append((ply_file, label_file))
            else:
                print(f"Warning: No label file found for {ply_file}")
        
        print(f"Valid samples: {len(self.valid_samples)}")
        
        if len(self.valid_samples) == 0:
            raise ValueError("No valid samples found! Please check your data directories.")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        ply_file, label_file = self.valid_samples[idx]
        
        # Load point cloud
        ply_path = os.path.join(self.ply_dir, ply_file)
        points = load_ply_file(ply_path)
        
        # Only use XYZ coordinates (first 3 channels)
        points_xyz = points[:, :3]
        
        # Load labels (all wires)
        label_path = os.path.join(self.label_dir, label_file)
        bboxes = load_label_file(label_path)
        
        # Normalize point cloud
        points_xyz, centroid, scale = normalize_point_cloud(points_xyz)
        
        # Apply augmentation if training
        if self.augmentation and self.split == 'train':
            points_xyz = rotate_point_cloud(points_xyz, angle_range=15)
            points_xyz = jitter_point_cloud(points_xyz, sigma=0.01, clip=0.05)
            points_xyz = random_flip_point_cloud(points_xyz)
            points_xyz = random_scale_point_cloud(points_xyz, scale_low=0.9, scale_high=1.1)
        
        # Sample fixed number of points
        points_xyz = sample_points(points_xyz, self.num_points)
        
        # Prepare labels for multiple wires
        cls_labels = np.zeros(self.max_wires, dtype=np.float32)
        bbox_params = np.zeros((self.max_wires, 7), dtype=np.float32)
        
        # Fill in actual wire labels
        num_wires = min(len(bboxes), self.max_wires)
        for i in range(num_wires):
            bbox = bboxes[i]
            
            # Normalize bbox center and size
            bbox_center = (bbox['center'] - centroid) / (scale + 1e-8)
            bbox_size = bbox['size'] / (scale + 1e-8)
            bbox_rotation = bbox['rotation']
            
            cls_labels[i] = 1.0  # Wire present
            bbox_params[i] = np.concatenate([bbox_center, bbox_size, [bbox_rotation]])
        
        # Convert to tensors
        points_tensor = torch.from_numpy(points_xyz).float()
        cls_labels_tensor = torch.from_numpy(cls_labels).float()
        bbox_tensor = torch.from_numpy(bbox_params).float()
        
        return {
            'points': points_tensor,
            'labels': cls_labels_tensor,  # [max_wires]
            'bboxes': bbox_tensor,  # [max_wires, 7]
            'filename': ply_file,
            'num_wires': num_wires
        }


def create_dataloaders_multi(config, train_split=0.8):
    """
    Create train and validation dataloaders for multi-wire detection.
    
    Args:
        config: Configuration dictionary
        train_split: Ratio of training data
    
    Returns:
        train_loader, val_loader
    """
    ply_dir = config['data']['ply_dir']
    label_dir = config['data']['label_dir']
    num_points = config['data']['num_points']
    batch_size = config['training']['batch_size']
    num_workers = config['hardware']['num_workers']
    max_wires = config['model'].get('max_wires', 6)
    
    # Get list of all samples to determine split
    full_dataset_temp = WireDetectionMultiDataset(
        ply_dir=ply_dir,
        label_dir=label_dir,
        num_points=num_points,
        augmentation=False,
        split='train',
        max_wires=max_wires
    )
    
    # Split indices
    total_samples = len(full_dataset_temp)
    train_size = int(train_split * total_samples)
    val_size = total_samples - train_size
    
    # Set random seed for reproducibility
    torch.manual_seed(config['data']['random_seed'])
    
    # Get split indices
    indices = torch.randperm(total_samples).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create separate datasets for train and val with different augmentation settings
    train_dataset = WireDetectionMultiDataset(
        ply_dir=ply_dir,
        label_dir=label_dir,
        num_points=num_points,
        augmentation=True,  # Enable augmentation for training
        split='train',
        max_wires=max_wires
    )
    
    val_dataset = WireDetectionMultiDataset(
        ply_dir=ply_dir,
        label_dir=label_dir,
        num_points=num_points,
        augmentation=False,  # Disable augmentation for validation
        split='val',
        max_wires=max_wires
    )
    
    # Create subset samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # Create dataloaders with samplers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=config['hardware']['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=config['hardware']['pin_memory'],
        drop_last=False
    )
    
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Max wires per sample: {max_wires}")
    
    return train_loader, val_loader