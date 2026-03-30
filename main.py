import os
import time
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import json
import csv
from timm.utils import accuracy, AverageMeter
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
import argparse

# Local imports
from models import build_model, get_model_config
from data_prep import build_loader
from utils import create_logger

# Base Configuration with all possible hyperparameters
BASE_CONFIG = {
    'DATA': {
        'DATA_PATH': 'food_subset',
        'BATCH_SIZE': 64,
        'NUM_WORKERS': 2,
        'PIN_MEMORY': True,
    },
    'MODEL': {
        'NAME': None, # Must be specified by model config
        'NUM_CLASSES': 5,
        'DROP_PATH_RATE': 0.1,
        'LABEL_SMOOTHING': 0.1,
    },
    'TRAIN': {
        'START_EPOCH': 0,
        'EPOCHS': 300,
        'BASE_LR': 5e-4,
        'WEIGHT_DECAY': 0.05,
        'CLIP_GRAD': 5.0,
        'WARMUP_EPOCHS': 5,
        'WARMUP_LR': 1e-6,
        'MIN_LR': 1e-5,
        'OPT': 'adamw',
        'SCHED': 'cosine',
    },
    'OUTPUT': 'outputs',
    'AMP_ENABLE': True,
}

class Config:
    def __init__(self, model_config):
        # Merge model_config into BASE_CONFIG
        for section, params in model_config.items():
            if isinstance(params, dict) and section in BASE_CONFIG and isinstance(BASE_CONFIG[section], dict):
                for k, v in params.items():
                    if v is not None:
                        BASE_CONFIG[section][k] = v
            elif params is not None:
                BASE_CONFIG[section] = params
        
        # Set attributes from merged config
        self.__dict__.update(BASE_CONFIG)
        
        # Convert dicts to simple objects for dot notation
        self.DATA = type('', (), self.DATA)()
        self.MODEL = type('', (), self.MODEL)()
        self.TRAIN = type('', (), self.TRAIN)()
        
        # Specific output dir
        self.OUTPUT = os.path.join(self.OUTPUT, self.MODEL.NAME)
        os.makedirs(self.OUTPUT, exist_ok=True)

    def defrost(self): pass
    def freeze(self): pass

def main(config, logger):
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config)
    
    logger.info(f"Creating model: {config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    
    optimizer = create_optimizer(argparse_namespace(opt=config.TRAIN.OPT, lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY), model)
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP_ENABLE)
    
    # Scheduler setup
    lr_scheduler, _ = create_scheduler(argparse_namespace(
        sched=config.TRAIN.SCHED, 
        epochs=config.TRAIN.EPOCHS, 
        warmup_epochs=config.TRAIN.WARMUP_EPOCHS, 
        warmup_lr=config.TRAIN.WARMUP_LR, 
        min_lr=config.TRAIN.MIN_LR, 
        cooldown_epochs=0
    ), optimizer)

    # Setup metrics logging to file
    log_file = os.path.join(config.OUTPUT, 'metrics.csv')
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    max_accuracy = 0.0
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_loss, train_acc = train_one_epoch(config, model, data_loader_train, optimizer, epoch, lr_scheduler, scaler, logger)
        
        val_acc, val_loss = validate(config, data_loader_val, model, logger)
        
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save metrics
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        max_accuracy = max(max_accuracy, val_acc)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler, scaler, logger):
    model.train()
    optimizer.zero_grad()
    
    num_steps = len(data_loader)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    for idx, (samples, targets) in enumerate(data_loader):
        samples, targets = samples.cuda(non_blocking=True), targets.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
            loss = torch.nn.CrossEntropyLoss()(outputs, targets)

        scaler.scale(loss).backward()
        if config.TRAIN.CLIP_GRAD:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        lr_scheduler.step_update(epoch * num_steps + idx)
        
        acc1, _ = accuracy(outputs, targets, topk=(1, 5))
        loss_meter.update(loss.item(), targets.size(0))
        acc_meter.update(acc1.item(), targets.size(0))

        if idx % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch: [{epoch}][{idx}/{num_steps}] lr {lr:.6f} loss {loss_meter.avg:.4f} acc {acc_meter.avg:.2f}%')

    return loss_meter.avg, acc_meter.avg

@torch.no_grad()
def validate(config, data_loader, model, logger=None):
    model.eval()
    acc1_meter, loss_meter = AverageMeter(), AverageMeter()

    for images, target in data_loader:
        images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)
        output = model(images)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        acc1, _ = accuracy(output, target, topk=(1, 5))

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))

    return acc1_meter.avg, loss_meter.avg

class argparse_namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def parse_option():
    parser = argparse.ArgumentParser('Unified Training Script', add_help=False)
    parser.add_argument('--model_to_run', type=str, required=True, help='Model name to run (e.g., as_mlp_tiny, deit_tiny, resnext50_local)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_option()
    MODEL_TO_RUN = args.model_to_run
    
    # Load model-specific config via models helper
    model_config_data = get_model_config(MODEL_TO_RUN)
    config = Config(model_config_data)
    
    torch.cuda.set_device(0)
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    logger = create_logger(output_dir=config.OUTPUT, name=config.MODEL.NAME)
    main(config, logger)