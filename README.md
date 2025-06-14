GeoLifeCLEF2025 Model Analysis

Methodology Overview

This notebook implements a Multi-Modal Ensemble Deep Learning approach for biodiversity prediction using three different data sources combined into a single unified model.

Data Sources & Preprocessing

1. Landsat Time Series Data

•
Format: Tensor cubes with shape [6, 4, 21]

•
Content: 6 spectral bands × 4 quarters × 21 years

•
Preprocessing:

•
NaN handling with torch.nan_to_num()

•
Tensor permutation from (C,H,W) to (H,W,C)

•
LayerNorm normalization



2. Bioclimatic Time Series Data

•
Format: Tensor cubes with shape [4, 19, 12]

•
Content: 4 climate variables × 19 years × 12 months

•
Preprocessing:

•
NaN handling with torch.nan_to_num()

•
Tensor permutation from (C,H,W) to (H,W,C)

•
LayerNorm normalization



3. Sentinel-2 Satellite Imagery

•
Format: TIFF files with 4 bands (RGB + NIR)

•
Content: 64×64 pixel patches

•
Preprocessing:

•
Quantile normalization (2nd-98th percentile)

•
Data augmentation (rotation, brightness/contrast)

•
Normalization with mean=(0.5,0.5,0.5,0.5), std=(0.5,0.5,0.5,0.5)



Model Architecture

MultimodalEnsemble Model

•
Base Architecture: Swin Transformer (swin_t)

•
Three Parallel Branches:

Branch 1: Landsat Processing

•
LayerNorm([6,4,21])

•
Swin Transformer (modified first conv layer: 6→96 channels)

•
Projection layer: 768→1000 features

Branch 2: Bioclimatic Processing

•
LayerNorm([4,19,12])

•
Swin Transformer (modified first conv layer: 4→96 channels)

•
Projection layer: 768→1000 features

Branch 3: Sentinel-2 Processing

•
Swin Transformer with ImageNet pre-trained weights

•
Modified first conv layer: 4→96 channels (for 4-band input)

•
Projection layer: 768→1000 features

Fusion & Classification

•
Feature Concatenation: 3000 features (1000 from each branch)

•
Classification Head:

•
Linear(3000→4096) + GELU + Dropout(0.1)

•
Linear(4096→num_classes) + GELU + Dropout(0.1)

•
Linear(num_classes→num_classes) - Final output



Training Configuration

Hyperparameters

•
Learning Rate: 8e-5

•
Epochs: 12

•
Batch Size: 256

•
Optimizer: AdamW

•
Scheduler: CosineAnnealingLR (T_max=25)

•
Loss Function: BCEWithLogitsLoss with positive weighting

Data Filtering

•
Species Selection: Only species with >5 occurrences (3,425 out of 5,016 species)

•
Multi-label Classification: Each survey can have multiple species

Inference & Post-processing

Prediction Strategy

1.
Sigmoid Activation: Convert logits to probabilities

2.
Adaptive Thresholding:

•
Primary threshold: 0.18

•
Fallback: Top 14 predictions if <14 species above threshold



3.
Species Mapping: Convert class indices back to original species IDs

Output Format

•
Submission File: CSV with surveyId and space-separated species predictions

•
Prediction Logic: Ensures minimum 14 species per survey

Key Technical Features

Advantages

1.
Multi-modal Fusion: Leverages complementary information from three data sources

2.
Pre-trained Backbone: Uses ImageNet pre-trained Swin Transformer for Sentinel-2

3.
Adaptive Architecture: Custom conv layers to handle different input dimensions

4.
Robust Preprocessing: Quantile normalization and NaN handling

5.
Flexible Thresholding: Adaptive prediction strategy

Architecture Innovations

•
Unified Swin Transformer: Same architecture for all three modalities

•
Feature Projection: Standardizes feature dimensions before fusion

•
Deep Classification Head: Multi-layer classifier for complex decision boundaries

•
Layer Normalization: Applied to time series data for stability

Data Flow Summary

Plain Text


Landsat [6,4,21] → LayerNorm → Swin-T → Proj → [1000]
                                                    ↓
Bioclim [4,19,12] → LayerNorm → Swin-T → Proj → [1000] → Concat → [3000] → Classifier → [3425]
                                                    ↓
Sentinel [64,64,4] → Quantile Norm → Swin-T → Proj → [1000]


This approach represents a sophisticated end-to-end multi-modal deep learning solution that effectively combines temporal, climatic, and visual information for biodiversity prediction.

