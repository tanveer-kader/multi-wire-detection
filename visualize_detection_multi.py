#!/usr/bin/env python3
"""
Enhanced visualization script for MULTI-WIRE detection with clear, annotated results.
Shows all 6 wires with RED for predictions and GREEN for ground truth.
"""

import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

from models.pointnet2_multi import PointNet2MultiDetector
from utils.preprocessing import load_ply_file, load_label_file, normalize_point_cloud, sample_points


def draw_3d_bbox(ax, center, size, rotation, color='red', label='Prediction', linewidth=2):
    """Draw a 3D bounding box."""
    w, h, d = size
    
    # Create corners of axis-aligned box
    corners = np.array([
        [-w/2, -h/2, -d/2], [w/2, -h/2, -d/2],
        [w/2, h/2, -d/2], [-w/2, h/2, -d/2],
        [-w/2, -h/2, d/2], [w/2, -h/2, d/2],
        [w/2, h/2, d/2], [-w/2, h/2, d/2]
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
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    
    # Draw edges
    for edge in edges:
        points_edge = corners[edge]
        ax.plot3D(*points_edge.T, color=color, linewidth=linewidth, label=label if edge == edges[0] else "")
    
    # Mark center
    ax.scatter(*center, color=color, s=100, marker='x', linewidths=3)
    
    return corners


def visualize_detection_enhanced(ply_path, model, config, device, save_path=None, show_ground_truth=True):
    """
    Create enhanced visualization with multiple views and annotations for ALL WIRES.
    """
    # Load point cloud
    points = load_ply_file(ply_path)
    points_xyz = points[:, :3]
    original_points = points_xyz.copy()
    
    # Load ground truth if available
    label_path = ply_path.replace('.ply', '.txt').replace('ply_files', 'labels')
    gt_bboxes = []
    if os.path.exists(label_path):
        try:
            gt_bboxes = load_label_file(label_path)
        except Exception as e:
            print(f"  Warning: Could not load labels: {e}")
    
    # Normalize and prepare for model
    points_xyz_norm, centroid, scale = normalize_point_cloud(points_xyz)
    points_sampled = sample_points(points_xyz_norm, config['data']['num_points'])
    points_tensor = torch.from_numpy(points_sampled).float().unsqueeze(0).to(device)
    
    # Predict ALL WIRES
    with torch.no_grad():
        cls_logits, bbox_preds = model(points_tensor)
    
    # Process predictions for each wire
    confidences = torch.sigmoid(cls_logits[0]).cpu().numpy()
    threshold = config['detection']['confidence_threshold']
    
    detected_wires = []
    for i in range(len(confidences)):
        if confidences[i] > threshold:
            bbox_params = bbox_preds[0, i].cpu().numpy()
            center = bbox_params[:3] * scale + centroid
            size = np.abs(bbox_params[3:6] * scale)
            rotation = bbox_params[6]
            
            detected_wires.append({
                'center': center,
                'size': size,
                'rotation': rotation,
                'confidence': confidences[i],
                'index': i + 1
            })
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # ===== Subplot 1: Top view (XY plane) =====
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(original_points[:, 0], original_points[:, 1], c=original_points[:, 2], 
                cmap='viridis', s=1, alpha=0.5)
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_title('Top View (XY)', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Draw ALL ground truth boxes (green dashed)
    if gt_bboxes and show_ground_truth:
        for gt in gt_bboxes:
            w, h, d = gt['size']
            angle_deg = np.degrees(gt['rotation'])
            rect_gt = Rectangle((gt['center'][0]-w/2, gt['center'][1]-h/2), w, h,
                               angle=angle_deg, linewidth=2, edgecolor='green',
                               facecolor='none', linestyle='--')
            ax1.add_patch(rect_gt)
            ax1.plot(gt['center'][0], gt['center'][1], 'g+', markersize=12, markeredgewidth=2)
    
    # Draw ALL predicted boxes (red solid)
    if detected_wires:
        for wire in detected_wires:
            w, h, d = wire['size']
            angle_deg = np.degrees(wire['rotation'])
            rect = Rectangle((wire['center'][0]-w/2, wire['center'][1]-h/2), w, h, 
                             angle=angle_deg, linewidth=3, edgecolor='red', 
                             facecolor='none')
            ax1.add_patch(rect)
            ax1.plot(wire['center'][0], wire['center'][1], 'rx', markersize=12, markeredgewidth=3)
    
    # Add legend only once
    if gt_bboxes:
        green_patch = mpatches.Patch(facecolor='none', edgecolor='green', linestyle='--', 
                                    linewidth=2, label=f'Ground Truth ({len(gt_bboxes)})')
        red_patch = mpatches.Patch(facecolor='none', edgecolor='red', 
                                  linewidth=3, label=f'Predicted ({len(detected_wires)})')
        ax1.legend(handles=[green_patch, red_patch], fontsize=10)
    
    # ===== Subplot 2: Side view (XZ plane) =====
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(original_points[:, 0], original_points[:, 2], c=original_points[:, 1],
                cmap='viridis', s=1, alpha=0.5)
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Z', fontsize=12)
    ax2.set_title('Side View (XZ)', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Draw ALL ground truth boxes
    if gt_bboxes and show_ground_truth:
        for gt in gt_bboxes:
            w, h, d = gt['size']
            rect_gt = Rectangle((gt['center'][0]-w/2, gt['center'][2]-d/2), w, d,
                               linewidth=2, edgecolor='green', facecolor='none', linestyle='--')
            ax2.add_patch(rect_gt)
            ax2.plot(gt['center'][0], gt['center'][2], 'g+', markersize=12, markeredgewidth=2)
    
    # Draw ALL predicted boxes
    if detected_wires:
        for wire in detected_wires:
            w, h, d = wire['size']
            rect = Rectangle((wire['center'][0]-w/2, wire['center'][2]-d/2), w, d,
                             linewidth=3, edgecolor='red', facecolor='none')
            ax2.add_patch(rect)
            ax2.plot(wire['center'][0], wire['center'][2], 'rx', markersize=12, markeredgewidth=3)
    
    # ===== Subplot 3: Front view (YZ plane) =====
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(original_points[:, 1], original_points[:, 2], c=original_points[:, 0],
                cmap='viridis', s=1, alpha=0.5)
    ax3.set_xlabel('Y', fontsize=12)
    ax3.set_ylabel('Z', fontsize=12)
    ax3.set_title('Front View (YZ)', fontsize=14, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # Draw ALL ground truth boxes
    if gt_bboxes and show_ground_truth:
        for gt in gt_bboxes:
            w, h, d = gt['size']
            rect_gt = Rectangle((gt['center'][1]-h/2, gt['center'][2]-d/2), h, d,
                               linewidth=2, edgecolor='green', facecolor='none', linestyle='--')
            ax3.add_patch(rect_gt)
            ax3.plot(gt['center'][1], gt['center'][2], 'g+', markersize=12, markeredgewidth=2)
    
    # Draw ALL predicted boxes
    if detected_wires:
        for wire in detected_wires:
            w, h, d = wire['size']
            rect = Rectangle((wire['center'][1]-h/2, wire['center'][2]-d/2), h, d,
                             linewidth=3, edgecolor='red', facecolor='none')
            ax3.add_patch(rect)
            ax3.plot(wire['center'][1], wire['center'][2], 'rx', markersize=12, markeredgewidth=3)
    
    # ===== Subplot 4: 3D view =====
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
                c=original_points[:, 2], cmap='viridis', s=1, alpha=0.3)
    
    # Draw ALL ground truth boxes in 3D
    if gt_bboxes and show_ground_truth:
        for gt in gt_bboxes:
            draw_3d_bbox(ax4, gt['center'], gt['size'], gt['rotation'],
                         color='green', label='', linewidth=2)
    
    # Draw ALL predicted boxes in 3D
    if detected_wires:
        for wire in detected_wires:
            draw_3d_bbox(ax4, wire['center'], wire['size'], wire['rotation'], 
                         color='red', label='', linewidth=3)
    
    ax4.set_xlabel('X', fontsize=10)
    ax4.set_ylabel('Y', fontsize=10)
    ax4.set_zlabel('Z', fontsize=10)
    ax4.set_title('3D View', fontsize=14, fontweight='bold')
    
    # ===== Subplot 5: Detection info =====
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')
    
    info_text = f"""
DETECTION RESULTS
{'='*40}

File: {os.path.basename(ply_path)}

Ground Truth Wires: {len(gt_bboxes)}
Detected Wires: {len(detected_wires)}
Match: {"✓ PERFECT" if len(detected_wires) == len(gt_bboxes) else "✗ MISMATCH"}

{'='*40}
PREDICTED WIRES:
{'='*40}
"""
    
    if detected_wires:
        for wire in detected_wires:
            info_text += f"""
Wire {wire['index']}:
  Confidence: {wire['confidence']:.1%}
  Center: ({wire['center'][0]:.2f}, {wire['center'][1]:.2f}, {wire['center'][2]:.2f})
  Size: ({wire['size'][0]:.2f}, {wire['size'][1]:.2f}, {wire['size'][2]:.2f})
  Rotation: {np.degrees(wire['rotation']):.1f}°
"""
    else:
        info_text += "\nNo wires detected!\n"
    
    ax5.text(0.05, 0.95, info_text, transform=ax5.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # ===== Subplot 6: Ground Truth Info =====
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    gt_text = f"""
GROUND TRUTH
{'='*40}

Total GT Wires: {len(gt_bboxes)}

{'='*40}
"""
    
    if gt_bboxes and show_ground_truth:
        for i, gt in enumerate(gt_bboxes, 1):
            gt_text += f"""
GT Wire {i}:
  Center: ({gt['center'][0]:.2f}, {gt['center'][1]:.2f}, {gt['center'][2]:.2f})
  Size: ({gt['size'][0]:.2f}, {gt['size'][1]:.2f}, {gt['size'][2]:.2f})
  Rotation: {np.degrees(gt['rotation']):.1f}°

"""
    else:
        gt_text += "\nNo ground truth!\n"
    
    # Add statistics
    if len(gt_bboxes) > 0:
        detection_rate = len(detected_wires) / len(gt_bboxes) * 100
        gt_text += f"""
{'='*40}
STATISTICS:
{'='*40}
  Detection Rate: {detection_rate:.1f}%
  True Positives: {min(len(detected_wires), len(gt_bboxes))}
  False Positives: {max(0, len(detected_wires) - len(gt_bboxes))}
  False Negatives: {max(0, len(gt_bboxes) - len(detected_wires))}
"""
    
    ax6.text(0.05, 0.95, gt_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Main title
    filename = os.path.basename(ply_path)
    match_status = "PERFECT ✓" if len(detected_wires) == len(gt_bboxes) else "MISMATCH ✗"
    color = 'green' if len(detected_wires) == len(gt_bboxes) else 'orange'
    fig.suptitle(f'{filename} - Detected {len(detected_wires)}/{len(gt_bboxes)} Wires - {match_status}',
                 fontsize=16, fontweight='bold', color=color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return detected_wires, gt_bboxes


def batch_visualize(checkpoint_path, ply_dir, output_dir, config_path='config_multi.yaml', num_samples=10):
    """Create enhanced visualizations for multiple samples."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    max_wires = config['model'].get('max_wires', 6)
    model = PointNet2MultiDetector(
        num_classes=config['model']['num_classes'],
        input_channels=config['model']['input_channels'],
        max_wires=max_wires
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f" Loaded model from epoch {checkpoint['epoch']}\n")
    
    # Get PLY files
    ply_files = sorted([f for f in os.listdir(ply_dir) if f.endswith('.ply')])[:num_samples]
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating enhanced multi-wire visualizations for {len(ply_files)} samples...\n")
    
    total_detected = 0
    total_gt = 0
    perfect_count = 0
    
    for i, ply_file in enumerate(ply_files, 1):
        print(f"[{i}/{len(ply_files)}] Processing {ply_file}...")
        ply_path = os.path.join(ply_dir, ply_file)
        save_path = os.path.join(output_dir, f"{os.path.splitext(ply_file)[0]}_multi_enhanced.png")
        
        try:
            detected, gt = visualize_detection_enhanced(ply_path, model, config, device, save_path)
            total_detected += len(detected)
            total_gt += len(gt)
            
            if len(detected) == len(gt):
                perfect_count += 1
                print(f"  ✓ Perfect: {len(detected)}/{len(gt)} wires")
            else:
                print(f"  ✗ Detected: {len(detected)}/{len(gt)} wires")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"BATCH VISUALIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total samples: {len(ply_files)}")
    print(f"Total wires detected: {total_detected}/{total_gt} ({100*total_detected/max(total_gt,1):.1f}%)")
    print(f"Perfect detections: {perfect_count}/{len(ply_files)} ({100*perfect_count/len(ply_files):.1f}%)")
    print(f"\nVisualizations saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Multi-Wire Detection Visualization')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_multi/best_model_multi.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to single PLY file or directory of PLY files')
    parser.add_argument('--output', type=str, default='results_multi/enhanced_viz',
                       help='Output directory for visualizations')
    parser.add_argument('--config', type=str, default='config_multi.yaml',
                       help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize (if input is directory)')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        # Single file
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        max_wires = config['model'].get('max_wires', 6)
        model = PointNet2MultiDetector(
            num_classes=config['model']['num_classes'],
            input_channels=config['model']['input_channels'],
            max_wires=max_wires
        ).to(device)
        
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        os.makedirs(args.output, exist_ok=True)
        save_path = os.path.join(args.output, 
                                os.path.splitext(os.path.basename(args.input))[0] + '_multi_enhanced.png')
        
        detected, gt = visualize_detection_enhanced(args.input, model, config, device, save_path)
        print(f"\n Detected {len(detected)}/{len(gt)} wires")
        print(f"Visualization saved to: {save_path}")
    else:
        # Directory
        batch_visualize(args.checkpoint, args.input, args.output, 
                       args.config, args.num_samples)