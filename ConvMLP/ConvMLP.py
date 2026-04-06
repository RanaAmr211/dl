"""
ConvMLP-S for Food-101 Classification
https://openaccess.thecvf.com/content/CVPR2023W/WFM/papers/Li_ConvMLP_Hierarchical_Convolutional_MLPs_for_Vision_CVPRW_2023_paper.pdf
"""
import torch
import torch.nn as nn
from timm.models import create_model

def convmlp_s(num_classes=5, **kwargs):
    """
    Load ConvMLP-S pre-trained from timm and fine-tune for Food-101.
    
    Args:
        num_classes: Number of output classes (default: 5 for subset, 101 for full)
        **kwargs: Additional hyperparameters from config (safely ignored for timm models)
    
    Returns:
        Modified ConvMLP-S model ready for training
    """
    # Extract parameters (ignored for timm pre-trained models)
    drop_path_rate = kwargs.pop('drop_path_rate', 0.1)
    
    # Load pre-trained ConvMLP-S from timm
    model = create_model('convmlp_s', pretrained=True, num_classes=num_classes)
    
    return model