"""
PVT-Tiny (Pyramid Vision Transformer Tiny)
Load from timm (PyTorch Image Models)
https://github.com/rwightman/pytorch-image-models
"""
import torch
import torch.nn as nn
from timm.models import create_model

def pvt_tiny(num_classes=5, **kwargs):
    """
    Load PVT-Tiny from timm pre-trained models.
    
    Args:
        num_classes: Number of output classes
        **kwargs: Additional hyperparameters from config (safely ignored)
    
    Returns:
        PVT-Tiny model ready for training
    """
    # Extract parameters (safely ignored if not present)
    _ = kwargs.pop('drop_path_rate', 0.1)
    
    # Remove any other unexpected kwargs
    for key in list(kwargs.keys()):
        kwargs.pop(key, None)
    
    # Load pre-trained PVT-Tiny from timm
    model = create_model('pvt_v2_b0', pretrained=True, num_classes=num_classes)
    
    return model