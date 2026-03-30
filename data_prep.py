import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler

def build_loader(config):
    # Unified Preprocessing
    train_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, "train"), transform=train_transform)
    eval_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, "validation"), transform=eval_transform)

    if dist_is_initialized():
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.DATA.BATCH_SIZE, 
        shuffle=(train_sampler is None),
        num_workers=config.DATA.NUM_WORKERS, 
        pin_memory=config.DATA.PIN_MEMORY, 
        sampler=train_sampler
    )
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=config.DATA.BATCH_SIZE, 
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS, 
        pin_memory=config.DATA.PIN_MEMORY, 
        sampler=eval_sampler
    )

    return train_dataset, eval_dataset, train_loader, eval_loader

def dist_is_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()
