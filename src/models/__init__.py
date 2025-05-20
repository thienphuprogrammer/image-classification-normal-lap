"""Models module for image classification.

This module provides access to optimized CNN architectures for image classification tasks.
All models support transfer learning, mixed precision training, and advanced regularization.
"""

__version__ = '1.0.0'

# Base model
from src.models.base_cnn import BaseCNN

# Optimized models
from src.models.efficient_cnn import EfficientCNN
from src.models.resnet_improved_cnn import ResNetImproved

# Export all models
__all__ = [
    'BaseCNN',
    'EfficientCNN',
    'ResNetImproved'
]