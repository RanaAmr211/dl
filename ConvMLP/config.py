def get_config():
    return {
        'MODEL': {
            'NAME': 'convmlp_s',
            'TAG': 'MLP',
            'NUM_CLASSES': 5,
            'DROP_PATH_RATE': 0.1,
            'LABEL_SMOOTHING': 0.1,
        },
        'DATA': {
            'DATA_PATH': None,
            'BATCH_SIZE': None,
            'NUM_WORKERS': None,
            'PIN_MEMORY': None,
        },
        'TRAIN': {
            'START_EPOCH': None,
            'EPOCHS': 100,
            'BASE_LR': 5e-4,
            'WEIGHT_DECAY': 1e-4,
            'CLIP_GRAD': 5.0,
            'WARMUP_EPOCHS': 10,
            'WARMUP_LR': 1e-6,
            'MIN_LR': 1e-5,
            'OPT': 'adamw',
            'SCHED': 'cosine',
        },
        'OUTPUT': None,
    }