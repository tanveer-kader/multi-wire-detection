import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """
    Calculate Euclidean distance between each two points.
    
    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).reshape(B, N, 1)
    dist += torch.sum(dst ** 2, -1).reshape(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling
    
    Args:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    
    Returns:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].reshape(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Query ball point
    
    Args:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    
    Returns:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).reshape(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].reshape(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def index_points(points, idx):
    """
    Index points with given indices
    
    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    
    Returns:
        new_points: indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).reshape(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Sample and group points
    
    Args:
        npoint: number of points to sample
        radius: ball query radius
        nsample: max samples per ball
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    
    Returns:
        new_xyz: sampled points position data, [B, npoint, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
    
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B, npoint, nsample]
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, 3]
    grouped_xyz_norm = grouped_xyz - new_xyz.reshape(B, S, 1, C)  # [B, npoint, nsample, 3]
    
    if points is not None:
        grouped_points = index_points(points, idx)  # [B, npoint, nsample, D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, 3+D]
    else:
        new_points = grouped_xyz_norm
    
    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Group all points (for global feature extraction)
    
    Args:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    
    Returns:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.reshape(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.reshape(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Args:
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, N, C]
        
        Returns:
            new_xyz: sampled points position data, [B, S, 3]
            new_points_concat: sample points feature data, [B, S, D']
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        
        # new_xyz: [B, npoint, 3], new_points: [B, npoint, nsample, C+3]
        new_points = new_points.permute(0, 3, 2, 1).contiguous()  # [B, C+3, nsample, npoint]
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        new_points = torch.max(new_points, 2)[0]  # [B, D', npoint]
        new_points = new_points.permute(0, 2, 1).contiguous()  # [B, npoint, D']
        
        return new_xyz, new_points


class PointNet2Backbone(nn.Module):
    def __init__(self, input_channels=3):
        super(PointNet2Backbone, self).__init__()
        
        # Set Abstraction layers
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32, 
            in_channel=input_channels + 3, mlp=[64, 64, 128],
            group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64, 
            in_channel=128 + 3, mlp=[128, 128, 256],
            group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, 
            in_channel=256 + 3, mlp=[256, 512, 1024],
            group_all=True
        )
    
    def forward(self, xyz, features=None):
        """
        Args:
            xyz: [B, N, 3]
            features: [B, N, C] or None
        
        Returns:
            global_features: [B, 1024]
        """
        B, N, _ = xyz.shape
        
        if features is None:
            features = xyz
        
        # SA layer 1
        l1_xyz, l1_points = self.sa1(xyz, features)
        
        # SA layer 2
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        
        # SA layer 3 (global features)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Global feature - l3_points is [B, 1, 1024]
        global_features = l3_points.squeeze(1)  # [B, 1024]
        
        return global_features


class PointNet2Detector(nn.Module):
    def __init__(self, num_classes=1, input_channels=3):
        super(PointNet2Detector, self).__init__()
        
        self.backbone = PointNet2Backbone(input_channels=input_channels)
        
        # Classification head (wire or not)
        self.cls_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Bounding box regression head (x, y, z, w, h, d, rotation)
        self.bbox_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 7)  # center(3) + size(3) + rotation(1)
        )
    
    def forward(self, xyz, features=None):
        """
        Args:
            xyz: [B, N, 3]
            features: [B, N, C] or None
        
        Returns:
            cls_logits: [B, num_classes]
            bbox_pred: [B, 7]
        """
        global_features = self.backbone(xyz, features)
        
        cls_logits = self.cls_head(global_features)
        bbox_pred = self.bbox_head(global_features)
        
        return cls_logits, bbox_pred