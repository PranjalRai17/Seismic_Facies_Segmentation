# Seismic Facies Segmentation (Netherlands F3)

## Overview
This project focuses on semantic segmentation of seismic data for identifying subsurface facies. A U-Net based convolutional neural network is trained on the Netherlands F3 dataset to learn geological structures directly from seismic slices.

The goal was not just to train a model, but to build a complete and reliable pipeline — including data handling, training strategy, evaluation, and inference — while avoiding common pitfalls such as data leakage.

---

## Problem Statement
Seismic interpretation is a critical task in geophysics and petroleum exploration. Manual interpretation is time-consuming and subjective. This project explores how deep learning can automate facies segmentation and capture meaningful geological patterns.

---

## Approach

### Data Handling
- Used inline seismic slices and corresponding segmentation masks
- Implemented a **spatial train-validation split** to prevent leakage between adjacent slices
- Applied preprocessing:
  - resizing (256×256)
  - normalization (mean-std scaling)

### Data Augmentation
- Horizontal and vertical flips
- Small rotations (±10°)
- Contrast scaling

Augmentation was carefully chosen to preserve geological realism.

---

### Model
- U-Net architecture (encoder-decoder with skip connections)
- Channel configuration: 48 → 96 → 192 → 384 → bottleneck
- Batch Normalization for stable training
- Dropout for regularization

---

### Loss Function
A combination of:
- **Focal Loss** (to focus on hard pixels and class imbalance)
- **Dice Loss** (to improve overlap and boundary quality)

This combination significantly improved validation performance compared to standard cross-entropy.

---

### Training Strategy
- Optimizer: Adam with weight decay
- Learning rate scheduling (ReduceLROnPlateau)
- Gradient clipping
- Early stopping based on validation loss

---

### Evaluation Metrics
- Mean Intersection over Union (IoU)
- Dice Score
- Pixel Accuracy

Validation was performed on spatially separated data to ensure realistic performance.

---

## Results

- Validation Loss: ~0.39  
- Mean IoU: ~0.6–0.75  
- Consistent segmentation of geological layers  
- Good boundary alignment with ground truth  

The model generalizes well across unseen seismic sections, indicating that it learned structural patterns rather than memorizing data.

---
## Key Learnings
Spatial data splitting is essential for seismic datasets to avoid leakage
Model capacity must be balanced — too large leads to overfitting, too small leads to underfitting
Loss function design (Focal + Dice) plays a major role in segmentation performance
Proper evaluation is more important than achieving low training loss
## Future Improvements
Try advanced architectures (UNet++, DeepLabV3+)
Incorporate 3D seismic volumes instead of 2D slices
Add test-time augmentation for more stable predictions
Build a lightweight web interface for visualization