import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.preprocessing import (
    load_ply_file, load_label_file, normalize_point_cloud,
    sample_points, rotate_point_cloud, jitter_point_cloud,
    random_flip_point_cloud, random_scale_point_cloud
)


class WireDetectionDataset(Dataset):
    def __init__(self, ply_dir, label_dir, num_points=8192, augmentation=True, split='train'):
        """
        Wire Detection Dataset
        
        Args:
            ply_dir: Directory containing .ply files
            label_dir: Directory containing .txt label files
            num_points: Number of points to sample
            augmentation: Whether to apply data augmentation
            split: 'train' or 'val'
        """
        self.ply_dir = ply_dir
        self.label_dir = label_dir
        self.num_points = num_points
        self.augmentation = augmentation
        self.split = split
        
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
        
        # Load labels
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
        
        # Prepare labels
        has_wire = len(bboxes) > 0
        
        if has_wire:
            # For now, use the first bounding box (extend later for multiple wires)
            bbox = bboxes[0]
            
            # Normalize bbox center and size
            bbox_center = (bbox['center'] - centroid) / (scale + 1e-8)
            bbox_size = bbox['size'] / (scale + 1e-8)
            bbox_rotation = bbox['rotation']
            
            # Combine: [cx, cy, cz, w, h, d, rotation]
            bbox_params = np.concatenate([bbox_center, bbox_size, [bbox_rotation]])
        else:
            # No wire detected
            bbox_params = np.zeros(7)
        
        # Convert to tensors
        points_tensor = torch.from_numpy(points_xyz).float()
        label_tensor = torch.tensor([1.0 if has_wire else 0.0], dtype=torch.float32)
        bbox_tensor = torch.from_numpy(bbox_params).float()
        
        return {
            'points': points_tensor,
            'label': label_tensor,
            'bbox': bbox_tensor,
            'filename': ply_file
        }


def create_dataloaders(config, train_split=0.8):
    """
    Create train and validation dataloaders.
    
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
    
    # Get list of all samples to determine split
    full_dataset_temp = WireDetectionDataset(
        ply_dir=ply_dir,
        label_dir=label_dir,
        num_points=num_points,
        augmentation=False,
        split='train'
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
    train_dataset = WireDetectionDataset(
        ply_dir=ply_dir,
        label_dir=label_dir,
        num_points=num_points,
        augmentation=True,  # Enable augmentation for training
        split='train'
    )
    
    val_dataset = WireDetectionDataset(
        ply_dir=ply_dir,
        label_dir=label_dir,
        num_points=num_points,
        augmentation=False,  # Disable augmentation for validation
        split='val'
    )
    
    # Create subset samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # Create dataloaders with samplers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,  # Use sampler instead of shuffle
        num_workers=num_workers,
        pin_memory=config['hardware']['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,  # Use sampler instead of shuffle
        num_workers=num_workers,
        pin_memory=config['hardware']['pin_memory'],
        drop_last=False
    )
    
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing dataset...")
    
    config = {
        'data': {
            'ply_dir': 'data/ply_files',
            'label_dir': 'data/labels',
            'num_points': 8192,
            'random_seed': 42
        },
        'training': {
            'batch_size': 4
        },
        'hardware': {
            'num_workers': 2,
            'pin_memory': True
        }
    }
    
    # Uncomment to test if you have data
    # train_loader, val_loader = create_dataloaders(config)
    # 
    # for batch in train_loader:
    #     print(f"Points shape: {batch['points'].shape}")
    #     print(f"Labels shape: {batch['label'].shape}")
    #     print(f"BBox shape: {batch['bbox'].shape}")
    #     break
    
    print("Dataset module ready!")