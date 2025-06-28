"""
Configuration for fast training with small 10-fruit dataset
"""

import os
from pathlib import Path

# Dataset paths
DATA_DIR = Path("data/small_dataset")
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Test"
CLASS_MAPPING_FILE = DATA_DIR / "class_mapping.json"

# Model parameters for fast training
IMG_SIZE = 64  # Smaller image size for faster training
BATCH_SIZE = 32
EPOCHS = 20  # Fewer epochs for fast training
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Model architecture (simplified for fast training)
MODEL_CONFIG = {
    'conv_layers': [
        {'filters': 32, 'kernel_size': 3, 'activation': 'relu'},
        {'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
        {'filters': 128, 'kernel_size': 3, 'activation': 'relu'},
    ],
    'dense_layers': [
        {'units': 128, 'activation': 'relu', 'dropout': 0.5},
        {'units': 64, 'activation': 'relu', 'dropout': 0.3},
    ],
    'num_classes': 10
}

# Training settings
TRAIN_CONFIG = {
    'optimizer': 'adam',
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy'],
    'callbacks': ['early_stopping', 'reduce_lr'],
    'early_stopping_patience': 5,
    'reduce_lr_patience': 3,
    'reduce_lr_factor': 0.5
}

# Data augmentation for small dataset
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# Output paths
MODEL_SAVE_DIR = Path("models")
MODEL_NAME = "fast_small_model"
LOG_DIR = Path("logs/fast_small")

# Ensure directories exist
MODEL_SAVE_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True) 