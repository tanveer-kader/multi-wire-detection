# Wire Detection in Point Clouds using PointNet++

Deep learning-based detection of wires in 3D point cloud data from circuit boards using PointNet++ architecture.

## Overview

This project detects wires in point cloud data using PointNet++. 

---

## Project Structure
```
multi-wire-detection/
├── data/
│   ├── ply_files/              # Point cloud files (.ply)
│   └── labels/                 # Bounding box labels (.txt)
│
├── models/
│   ├── pointnet.py             # PointNet++ single wire detector
│   └── pointnet_multi.py       # PointNet++ multi-wire detector
│
├── utils/
│   ├── preprocessing.py        # Data loading utilities
│   ├── data_loader.py          # PyTorch Dataset
│   ├── losses.py               # Loss functions
│   ├── metrics.py              # Evaluation metrics
│   └── visualization.py        # Visualization tools
│
├── train.py                    # Multi wire training
├── visualize_detection_multi.py  # Multi-wire visualization
│
├── config.yaml                 # Multi wire config
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/tanveer-kader/multi-wire-detection.git
cd multi-wire-detection
```

### 2. Create Virtual Environment
```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Or using conda
conda create -n multi-wire-detection python=3.12.3
conda activate multi-wire-detection
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Dataset Setup

### Directory Structure
```
data/
├── ply_files/
│   ├── circuit_001.ply
│   ├── circuit_002.ply
│   └── ... (111 files)
└── labels/
    ├── circuit_001.txt
    ├── circuit_002.txt
    └── ... (111 files)
```

#### Multi-Wire Detection (6 wires)

Train model to detect all 6 wires simultaneously:
```bash
python train.py --config config.yaml
```

### Evaluation

#### Evaluate Multi-Wire Model
```bash
python evaluate.py \
    --checkpoint checkpoints_multi/best_model.pth \
    --data_dir data/ply_files \
    --label_dir data/labels \
    --config config.yaml
```

### Visualization

#### Enhanced Multi-Wire Visualization

Visualize detection results with comprehensive views:

**Batch of files:**
```bash
python visualize_detection_multi.py \
    --checkpoint checkpoints_multi/best_model_multi.pth \
    --input data/ply_files/ \
    --output results/visualization \
    --config config.yaml \
    --num_samples 20
```

**PointNet++ Paper:**
```bibtex
@inproceedings{qi2017pointnetplusplus,
  title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
  author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}
```