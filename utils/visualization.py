import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def visualize_point_cloud(points, title="Point Cloud", save_path=None):
    """
    Visualize point cloud using matplotlib.
    
    Args:
        points: [N, 3] numpy array
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=points[:, 2], cmap='viridis', marker='o', s=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Equal aspect ratio
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_detection(points, pred_bbox=None, gt_bbox=None, save_path=None):
    """
    Visualize point cloud with predicted and ground truth bounding boxes.
    
    Args:
        points: [N, 3] numpy array
        pred_bbox: dict with 'center', 'size', 'rotation' (predicted)
        gt_bbox: dict with 'center', 'size', 'rotation' (ground truth)
        save_path: Path to save visualization
    """
    # Use matplotlib instead of Open3D for better compatibility
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot point cloud
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=points[:, 2], cmap='viridis', marker='o', s=1, alpha=0.3)
    
    # Helper function to draw bbox
    def draw_bbox(center, size, rotation, color, label):
        w, h, d = size
        # Create corners
        corners = np.array([
            [-w/2, -h/2, -d/2],
            [w/2, -h/2, -d/2],
            [w/2, h/2, -d/2],
            [-w/2, h/2, -d/2],
            [-w/2, -h/2, d/2],
            [w/2, -h/2, d/2],
            [w/2, h/2, d/2],
            [-w/2, h/2, d/2]
        ])
        
        # Rotation matrix around Z-axis
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        R = np.array([
            [cos_r, -sin_r, 0],
            [sin_r, cos_r, 0],
            [0, 0, 1]
        ])
        
        # Rotate and translate
        corners = (R @ corners.T).T + center
        
        # Define edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical
        ]
        
        # Draw edges
        for edge in edges:
            points_edge = corners[edge]
            ax.plot3D(*points_edge.T, color=color, linewidth=2)
        
        # Add label
        ax.text(center[0], center[1], center[2], label, color=color, fontsize=10)
    
    # Draw ground truth bbox (green)
    if gt_bbox is not None:
        draw_bbox(gt_bbox['center'], gt_bbox['size'], gt_bbox['rotation'], 
                  'green', 'GT')
    
    # Draw predicted bbox (red)
    if pred_bbox is not None:
        draw_bbox(pred_bbox['center'], pred_bbox['size'], pred_bbox['rotation'], 
                  'red', 'Pred')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Wire Detection')
    
    # Equal aspect ratio
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(history, save_dir='results'):
    """
    Plot training history (loss curves).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', etc.
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot classification accuracy if available
    if 'train_acc' in history:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        if 'val_acc' in history:
            plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Classification Accuracy')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training history plot to {save_path}")
    plt.close()


def visualize_batch_predictions(points_batch, pred_bboxes, gt_bboxes, 
                                 save_dir='results', batch_idx=0, num_samples=4):
    """
    Visualize predictions for a batch of samples.
    
    Args:
        points_batch: [B, N, 3] tensor
        pred_bboxes: [B, 7] tensor (predicted)
        gt_bboxes: [B, 7] tensor (ground truth)
        save_dir: Directory to save visualizations
        batch_idx: Batch index for naming
        num_samples: Number of samples to visualize
    """
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = min(points_batch.shape[0], num_samples)
    
    for i in range(batch_size):
        points = points_batch[i].cpu().numpy()
        pred_bbox_params = pred_bboxes[i].cpu().numpy()
        gt_bbox_params = gt_bboxes[i].cpu().numpy()
        
        # Convert to bbox dict format
        pred_bbox = {
            'center': pred_bbox_params[:3],
            'size': pred_bbox_params[3:6],
            'rotation': pred_bbox_params[6]
        }
        
        gt_bbox = {
            'center': gt_bbox_params[:3],
            'size': gt_bbox_params[3:6],
            'rotation': gt_bbox_params[6]
        }
        
        save_path = os.path.join(save_dir, f'batch{batch_idx}_sample{i}.png')
        
        # Use matplotlib for visualization
        visualize_detection(points, pred_bbox=pred_bbox, gt_bbox=gt_bbox, 
                          save_path=save_path)


if __name__ == "__main__":
    # Test visualization
    print("Testing visualization functions...")
    
    # Create dummy data
    points = np.random.randn(1000, 3)
    
    # Test point cloud visualization
    print("Testing point cloud visualization...")
    # visualize_point_cloud(points, title="Test Point Cloud")
    
    # Test bbox visualization
    bbox_gt = {
        'center': np.array([0, 0, 0]),
        'size': np.array([0.3, 0.2, 0.5]),
        'rotation': 0.5
    }
    
    bbox_pred = {
        'center': np.array([0.1, 0.05, 0.02]),
        'size': np.array([0.32, 0.18, 0.48]),
        'rotation': 0.45
    }
    
    print("Testing detection visualization...")
    # visualize_detection(points, pred_bbox=bbox_pred, gt_bbox=bbox_gt)
    
    print("Visualization functions ready!")