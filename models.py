from AS_MLP.AS_MLP import as_mlp_tiny
from DeiT.DeiT import deit_tiny_distilled_patch16_224, deit_tiny_patch16_224
from ResNeXt.ResNeXt import resnext50_32x4d

def get_model_config(model_name):
    if model_name == 'as_mlp_tiny':
        from AS_MLP.config import get_config
    elif model_name == 'deit_tiny':
        from DeiT.config import get_config
    elif model_name == 'resnext50_local':
        from ResNeXt.config import get_config
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return get_config()

def build_model(config):
    model_name = config.MODEL.NAME
    num_classes = config.MODEL.NUM_CLASSES
    
    # 1. Local Models
    if model_name == 'as_mlp_tiny':
        return as_mlp_tiny(num_classes=num_classes)
    elif model_name == 'deit_tiny':
        return deit_tiny_patch16_224(num_classes=num_classes)
    elif model_name == 'resnext50_local':
        return resnext50_32x4d() # num_classes already 5 after update

    # continue defining other models
