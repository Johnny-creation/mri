# Brain MRI Stroke Lesion Detection and Classification

A deep learning-based multi-task classification system for analyzing brain MRI scans to detect stroke lesions and classify their temporal stage using dual-modality imaging (DWI + FLAIR).

## Overview

This project implements a sophisticated neural network architecture for automated stroke analysis from brain MRI scans, performing two critical tasks:

1. **Lesion Detection**: Binary classification to identify the presence of stroke lesions
2. **Time Classification**: Binary classification to determine stroke timing stage (early vs. late, threshold: 270 minutes)

The system leverages multi-task learning with shared feature extraction and task-specific classification heads, enhanced by advanced attention mechanisms for improved performance on medical imaging data.

## Key Features

### Advanced Architecture
- **Backbone**: EfficientNet-B0 pre-trained on ImageNet for transfer learning
- **Dual-Modality Input**: Combines DWI (Diffusion-Weighted Imaging) and FLAIR (Fluid-Attenuated Inversion Recovery) sequences
- **Multi-Scale Feature Fusion**: Integrates mid-level and high-level features for richer representations
- **SCAE Attention**: Squeeze-and-Channel Attention Enhancement block with adaptive 1D convolution
- **Spatial Attention**: 7×7 convolutional attention for spatial feature refinement
- **DropBlock Regularization**: Advanced dropout technique for improved CNN regularization

### Medical Imaging Specific
- **DICOM Support**: Native handling of medical imaging DICOM format
- **Dual-Channel Processing**: Fuses complementary DWI and FLAIR modalities
- **Clinical Workflow**: Cascaded classification (time analysis only for lesion-positive cases)
- **Robust Preprocessing**: Min-max normalization and standardization for medical images

### Training & Evaluation
- **Multi-Task Learning**: Joint optimization of lesion detection and time classification
- **Comprehensive Metrics**: Accuracy, AUC-ROC, F1-score, sensitivity, specificity, PPV, NPV
- **Automatic Checkpointing**: Saves best model based on validation AUC
- **Detailed Logging**: Timestamped training logs with epoch-wise metrics

## Project Structure

```
mri/
├── main.py           # Entry point with CLI arguments
├── model.py          # Neural network architectures (SCAEBlock, SpatialAttention, MultiTaskModel)
├── dataset.py        # Data loading, preprocessing, and DICOM handling
├── train.py          # Training loop with multi-task loss
├── evaluate.py       # Comprehensive evaluation metrics
├── utils.py          # Utility functions (reproducibility, seed setting)
└── output/           # Training outputs
    ├── model_best.pt              # Best model checkpoint (highest val AUC)
    ├── model_final.pt             # Final epoch model
    └── training_log_*.txt         # Detailed training logs
```

## Architecture Details

### Model Pipeline

```
Input: 2-channel (DWI + FLAIR) 224×224 images
           ↓
EfficientNet-B0 Feature Extraction
           ↓
Multi-Scale Feature Fusion (layer 3: 40ch + final: 1280ch)
           ↓
Dimension Reduction (40→128, 1280→128)
           ↓
Concatenation (256 channels)
           ↓
SCAE Channel Attention (adaptive kernel)
           ↓
Spatial Attention (7×7 conv)
           ↓
DropBlock2D Regularization
           ↓
Global Average Pooling
           ↓
    ┌─────────────┴─────────────┐
    ↓                           ↓
Lesion Head                Time Head
(256→64→2)                 (256→64→2)
```

### SCAE Block (Squeeze-and-Channel Attention Enhancement)

The SCAE block is a key innovation in this implementation, replacing traditional Squeeze-and-Excitation (SE) blocks:

- **Adaptive Kernel Sizing**: k = |log₂(C) / γ + b|_odd (γ=2, b=1)
- **1D Convolution**: More parameter-efficient than fully connected layers
- **Local Cross-Channel Interaction**: Captures dependencies without global operations
- **Better Scalability**: Adapts to different feature dimensions automatically

### Multi-Task Learning Strategy

- **Shared Backbone**: Common feature extraction for both tasks
- **Task-Specific Heads**: Specialized classifiers for lesion detection and time classification
- **Weighted Loss**: L_total = L_lesion + λ × L_time (default λ=1.0)
- **Cascaded Evaluation**: Time classification metrics computed only on lesion-positive predictions

## Installation

### Requirements

```bash
pip install torch torchvision
pip install pydicom numpy pillow scikit-learn
```

### Dependencies
- Python 3.7+
- PyTorch 1.9+
- torchvision
- pydicom (for DICOM file reading)
- numpy
- Pillow (PIL)
- scikit-learn (for metrics)

## Dataset Preparation

### Expected Directory Structure

```
data/lesion-selected/
├── train/
│   ├── case_001_120min/
│   │   ├── DWI/
│   │   │   ├── slice_001.dcm
│   │   │   ├── slice_002.dcm
│   │   │   └── ...
│   │   └── FLAIR/
│   │       ├── slice_001.dcm
│   │       ├── slice_002.dcm
│   │       └── ...
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### Naming Conventions

- **Case Folders**: `{case_name}_{time_in_minutes}min/`
  - Time value determines the time classification label (≥270min → class 1, <270min → class 0)
- **Slice Files**: Must contain 'x' in filename for lesion-positive samples
  - Example: `slice_001_x.dcm` (has lesion), `slice_002.dcm` (no lesion)

### Data Requirements

- **Format**: DICOM (.dcm) files
- **Modalities**: Both DWI and FLAIR sequences required for each slice
- **Matching Slices**: DWI and FLAIR must have corresponding slices with same numbering
- **Image Size**: Any size (automatically resized to 224×224)

## Usage

### Basic Training

```bash
python main.py
```

### Training with Custom Parameters

```bash
python main.py \
    --base_path ../data/lesion-selected \
    --batch_size 32 \
    --epochs 70 \
    --lr 0.0008 \
    --dropout_rate 0.4 \
    --weight_decay 0.0001 \
    --time_loss_weight 1.0 \
    --seed 42
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--base_path` | str | `../data/lesion-selected` | Path to dataset root directory |
| `--batch_size` | int | 32 | Training batch size |
| `--epochs` | int | 70 | Number of training epochs |
| `--lr` | float | 0.0008 | Learning rate for Adam optimizer |
| `--dropout_rate` | float | 0.4 | Dropout probability in classifier heads |
| `--weight_decay` | float | 0.0 | L2 regularization weight |
| `--time_loss_weight` | float | 1.0 | Weight for time classification loss (λ) |
| `--seed` | int | 42 | Random seed for reproducibility |

### Training Output

The training process creates an `output/` directory with:
- `model_best.pt`: Best model based on validation time classification AUC
- `model_final.pt`: Model from the final epoch
- `training_log_YYYYMMDD_HHMMSS.txt`: Detailed log with metrics for each epoch



## Evaluation Metrics

The system provides comprehensive evaluation for both tasks:

### Lesion Detection Metrics
- **Accuracy**: Overall correct predictions
- **AUC-ROC**: Area under receiver operating characteristic curve
- **F1-Score**: Harmonic mean of precision and recall
- **Sensitivity (Recall)**: True positive rate
- **Specificity**: True negative rate
- **PPV**: Positive predictive value (precision)
- **NPV**: Negative predictive value

### Time Classification Metrics
Same metrics as above, but **evaluated only on samples predicted as lesion-positive**, reflecting the clinical workflow where time classification is only relevant for confirmed lesions.

## Technical Highlights

### 1. Multi-Scale Feature Fusion
Combines features from different depths of EfficientNet-B0:
- **Mid-level features** (layer 3, 40 channels): Captures fine-grained spatial details
- **High-level features** (final layer, 1280 channels): Captures semantic information
- Both are dimensionally reduced to 128 channels and concatenated

### 2. Attention Mechanisms

**SCAE (Channel Attention)**:
- Adaptively determines convolution kernel size based on channel count
- More efficient than SE block's fully connected layers
- Preserves local cross-channel dependencies

**Spatial Attention**:
- Applies 7×7 convolution on channel-pooled features
- Highlights important spatial regions
- Complements channel attention for comprehensive feature refinement

### 3. DropBlock Regularization
- Drops contiguous 7×7 regions instead of individual pixels
- More effective than standard dropout for CNNs
- Drop probability: 0.1
- Prevents overfitting while preserving spatial structure

### 4. Reproducibility
- Comprehensive seed setting for PyTorch, NumPy, random, and CUDA
- Deterministic CUDNN operations
- All randomness sources controlled via `--seed` argument

## Clinical Significance

### Why Dual-Modality (DWI + FLAIR)?
- **DWI**: Highly sensitive to acute ischemic changes (water diffusion restriction)
- **FLAIR**: Effective for visualizing both acute and chronic lesions, suppresses CSF signal
- **Combined**: Provides complementary information for comprehensive stroke assessment

### Why Multi-Task Learning?
- Time classification logically depends on lesion presence
- Shared features capture stroke-related patterns useful for both tasks
- More efficient than training separate models
- Encourages learning of robust, generalizable features

### Clinical Workflow Integration
The cascaded evaluation (time classification only on lesion-positive cases) mirrors clinical practice where temporal staging is only relevant after lesion identification.

## Future Improvements

Potential areas for enhancement:
- [ ] 3D volumetric processing instead of slice-by-slice
- [ ] Multi-class time classification (multiple time windows)
- [ ] Uncertainty quantification for clinical decision support
- [ ] Attention visualization for interpretability
- [ ] Cross-validation for robust performance estimation
- [ ] Ensemble methods for improved accuracy
- [ ] Data augmentation strategies for medical images
- [ ] External validation on different datasets

