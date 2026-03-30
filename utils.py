import os
import torch
import logging
from timm.utils import AverageMeter

def create_logger(output_dir, name):
    # Setup logging to both console and file
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Create formatters
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(output_dir, 'log.txt'), mode='a')
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)

    return logger
