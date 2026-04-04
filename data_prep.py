import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def build_loader(config):

    train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),          
    transforms.RandomHorizontalFlip(),   
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, "train"), transform=train_transform)
    eval_dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, "validation"), transform=eval_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.DATA.BATCH_SIZE, 
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS, 
        pin_memory=config.DATA.PIN_MEMORY,
    )
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=config.DATA.BATCH_SIZE, 
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS, 
        pin_memory=config.DATA.PIN_MEMORY,
    )

    return train_dataset, eval_dataset, train_loader, eval_loader
