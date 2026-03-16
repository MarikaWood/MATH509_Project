# Alzheimer’s Disease MRI Classification Using Convolutional Neural Networks

## Overview
This project develops a deep learning model to classify Alzheimer's disease severity from MRI brain scans. The goal is to explore how convolutional neural networks (CNNs) can be used for medical image classification tasks.

The model is trained using MRI images from the OASIS dataset and uses transfer learning with MobileNetV2 to classify images into three dementia categories.

---
## Dataset
Dataset: **OASIS MRI Dataset**

Source: Kaggle
https://www.kaggle.com/datasets/ninadaithal/imagesoasis

The dataset contains MRI brain scan image categorized into dementia stages:
- Non-Demented
- Very Mild Dementia
- Mild Dementia

Each MRI scan contains multiple slices per patient. To reduce redundancy and ensure consistent representation, specific slices (119–121) were selected for model training.

## Data Preprocessing
Several preprocessing steps were implemented to prepare the images for model training:
- MRI slices **119-121** were selected for each patient
- Randomly sampled images per patient to maintin dataset balance
- Resized images to **224 x 224 pixels**
- Converted images to grayscale
- Applied **image normalization**
- Implemented **data augmentation** including:
  - Rotation
  - Zoom
  - Brightness Adjustment
  - Horizontal flipping

A **patient-level train/test split (70/30)** was used to prevent data leakage between training and testing datasets.

## Model Architecture
The model uses **transfer learning with MobileNetV2** pretrained on ImageNet.

Key components:
- MobileNetV2 base model (frozen layers)
- Fine-tuning of final convolutional layers
- Additional convolutional layer
- Global average pooling
- Dense layer (256 units)
- Dropout for regularization
- Softmax output layer for 3-class classification

**Loss Function:** 
Categorical Cross-Entropy

**Optimizer:**
Adam

---

## Class Imbalance
To improve model performanc for underrepresented classes, the following approaches were used:
- Class weighting during model training
- Image augmentation to increase variability

