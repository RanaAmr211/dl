"""
DenseNet-121 for Food-101 Classification
Pre-trained on ImageNet from torchvision
https://arxiv.org/abs/1608.06993
"""
import torch
import torch.nn as nn
import torchvision.models as models

def densenet121(num_classes=5, **kwargs):
    """
    Load DenseNet-121 pre-trained on ImageNet and fine-tune for Food-101.
    
    Accepts hyperparameters from config and applies them to the model:
    - DROP_PATH_RATE controls dropout in the final classifier layer
    - Enables proper hyperparameter tuning via config files
    """
    # Extract dropout rate from config (follows the standard naming convention)
    drop_rate = kwargs.pop('drop_path_rate', 0.1)
    
    # Also support 'drop_rate' if specified
    if 'drop_rate' in kwargs:
        drop_rate = kwargs.pop('drop_rate', drop_rate)
    
    # Load pre-trained DenseNet-121
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    
    # Get classifier input dimension
    num_features = model.classifier.in_features
    
    # Build classifier with dropout regularization
    if drop_rate > 0:
        classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(num_features, num_classes)
        )
    else:
        classifier = nn.Linear(num_features, num_classes)
    
    # Replace ImageNet classifier with Food-101 classifier
    model.classifier = classifier
    
    return model
