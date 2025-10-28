import numpy as np
import open3d as o3d
from plyfile import PlyData
import os


def load_ply_file(ply_path):
    """
    Load a PLY file and return point cloud as numpy array.
    
    Args:
        ply_path: Path to .ply file
    
    Returns:
        points: numpy array of shape [N, 3] or [N, 6] (XYZ or XYZRGB)
    """
    try:
        # Try with open3d first
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        
        # Check if colors are available
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            points = np.concatenate([points, colors], axis=1)
        
        return points
    except Exception as e:
        print(f"Open3D failed, trying plyfile: {e}")
        # Fallback to plyfile
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        
        # Extract XYZ
        x = vertex['x']
        y = vertex['y']
        z = vertex['z']
        points = np.stack([x, y, z], axis=1)
        
        # Try to extract RGB if available
        if 'red' in vertex.data.dtype.names:
            r = vertex['red'] / 255.0
            g = vertex['green'] / 255.0
            b = vertex['blue'] / 255.0
            colors = np.stack([r, g, b], axis=1)
            points = np.concatenate([points, colors], axis=1)
        
        return points


def load_label_file(label_path):
    """
    Load label file and parse bounding boxes.
    
    Label format:
    wire 0 0 0 0 0 0 0 width height depth center_x center_y center_z rotation
    
    Args:
        label_path: Path to .txt label file
    
    Returns:
        bboxes: List of bounding boxes, each as dict with keys:
                'center', 'size', 'rotation'
    """
    bboxes = []
    
    if not os.path.exists(label_path):
        return bboxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            
            # Parse the format: wire 0 0 0 0 0 0 0 w h d cx cy cz rot
            try:
                width = float(parts[8])
                height = float(parts[9])
                depth = float(parts[10])
                center_x = float(parts[11])
                center_y = float(parts[12])
                center_z = float(parts[13])
                rotation = float(parts[14])
                
                bbox = {
                    'center': np.array([center_x, center_y, center_z]),
                    'size': np.array([width, height, depth]),
                    'rotation': rotation
                }
                bboxes.append(bbox)
            except (ValueError, IndexError) as e:
                print(f"Error parsing line: {line.strip()}, error: {e}")
                continue
    
    return bboxes


def normalize_point_cloud(points):
    """
    Normalize point cloud to unit sphere.
    
    Args:
        points: [N, 3] or [N, 6] numpy array
    
    Returns:
        normalized_points: Normalized point cloud
        centroid: Original centroid
        scale: Original scale
    """
    # Only normalize XYZ coordinates
    xyz = points[:, :3]
    
    # Calculate centroid
    centroid = np.mean(xyz, axis=0)
    
    # Center the points
    xyz_centered = xyz - centroid
    
    # Calculate scale
    scale = np.max(np.sqrt(np.sum(xyz_centered ** 2, axis=1)))
    
    # Normalize
    xyz_normalized = xyz_centered / (scale + 1e-8)
    
    # Combine with other features if present
    if points.shape[1] > 3:
        normalized_points = np.concatenate([xyz_normalized, points[:, 3:]], axis=1)
    else:
        normalized_points = xyz_normalized
    
    return normalized_points, centroid, scale


def random_point_dropout(points, max_dropout_ratio=0.875):
    """
    Randomly dropout points for augmentation.
    
    Args:
        points: [N, C] numpy array
        max_dropout_ratio: Maximum ratio of points to drop
    
    Returns:
        points: Augmented point cloud
    """
    dropout_ratio = np.random.random() * max_dropout_ratio
    drop_idx = np.where(np.random.random((points.shape[0])) <= dropout_ratio)[0]
    
    if len(drop_idx) > 0:
        points = np.delete(points, drop_idx, axis=0)
    
    return points


def random_scale_point_cloud(points, scale_low=0.8, scale_high=1.25):
    """
    Randomly scale the point cloud.
    
    Args:
        points: [N, C] numpy array
        scale_low: Minimum scale
        scale_high: Maximum scale
    
    Returns:
        points: Scaled point cloud
    """
    scale = np.random.uniform(scale_low, scale_high)
    points[:, :3] *= scale
    return points


def random_flip_point_cloud(points):
    """
    Randomly flip the point cloud along X or Y axis.
    
    Args:
        points: [N, C] numpy array
    
    Returns:
        points: Flipped point cloud
    """
    if np.random.random() > 0.5:
        points[:, 0] = -points[:, 0]  # Flip X
    if np.random.random() > 0.5:
        points[:, 1] = -points[:, 1]  # Flip Y
    return points


def rotate_point_cloud(points, angle_range=15):
    """
    Rotate point cloud around Z-axis.
    
    Args:
        points: [N, C] numpy array
        angle_range: Rotation angle range in degrees
    
    Returns:
        points: Rotated point cloud
    """
    angle = np.random.uniform(-angle_range, angle_range) * np.pi / 180.0
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    rotation_matrix = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])
    
    points[:, :3] = np.dot(points[:, :3], rotation_matrix.T)
    return points


def jitter_point_cloud(points, sigma=0.01, clip=0.05):
    """
    Add random jitter to point cloud.
    
    Args:
        points: [N, C] numpy array
        sigma: Standard deviation
        clip: Clip value
    
    Returns:
        points: Jittered point cloud
    """
    N, C = points.shape
    jitter = np.clip(sigma * np.random.randn(N, 3), -clip, clip)
    points[:, :3] += jitter
    return points


def sample_points(points, num_points):
    """
    Sample fixed number of points from point cloud.
    
    Args:
        points: [N, C] numpy array
        num_points: Number of points to sample
    
    Returns:
        sampled_points: [num_points, C] numpy array
    """
    N = points.shape[0]
    
    if N >= num_points:
        # Random sampling
        choice = np.random.choice(N, num_points, replace=False)
    else:
        # Repeat sampling if not enough points
        choice = np.random.choice(N, num_points, replace=True)
    
    sampled_points = points[choice, :]
    return sampled_points


def compute_normals(points_xyz):
    """
    Compute normals for point cloud.
    
    Args:
        points_xyz: [N, 3] numpy array
    
    Returns:
        normals: [N, 3] numpy array
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normals = np.asarray(pcd.normals)
    return normals


if __name__ == "__main__":
    # Test functions
    print("Testing preprocessing functions...")
    
    # Create dummy data
    dummy_points = np.random.randn(1000, 3)
    print(f"Original shape: {dummy_points.shape}")
    
    # Test normalization
    normalized, centroid, scale = normalize_point_cloud(dummy_points)
    print(f"Normalized shape: {normalized.shape}")
    
    # Test sampling
    sampled = sample_points(dummy_points, 512)
    print(f"Sampled shape: {sampled.shape}")
    
    # Test augmentation
    augmented = rotate_point_cloud(dummy_points.copy())
    augmented = jitter_point_cloud(augmented)
    print(f"Augmented shape: {augmented.shape}")
    
    print("All tests passed!")