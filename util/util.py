import time
import os
import logging
from logging.config import dictConfig
import torch
import random
import numpy as np
import os.path as osp

logging_config = dict(
    version=1,
    formatters={
        'f_t': {'format': '%(asctime)s | %(levelname)s | %(name)s %(message)s'}

    },
    handlers={
        'stream_handler': {
            'class': 'logging.StreamHandler',
            'formatter': 'f_t',
            'level': logging.INFO},
        'file_handler': {
            'class': 'logging.FileHandler',
            'formatter': 'f_t',
            'level': logging.INFO,
            'filename': None,
        }
    },
    root={
        'handlers': ['stream_handler', 'file_handler'],
        'level': logging.DEBUG,
    },
)


def create_logger(logdir):
    """Set up the logger for saving log file on the disk

    Args:
        cfg: configuration dict
        postfix: postfix of the log file name

    Return:
        logger: a logger for record essential information
    """
    # set up logger

    set_name = logdir.split('/')[2]
    log_file = f'{set_name}.log'
    log_file_path = os.path.join(logdir, log_file)
    logging_config['handlers']['file_handler']['filename'] = log_file_path
    open(log_file_path, 'w').close()  # Clear the content of logfile
    # get logger from dictConfig
    dictConfig(logging_config)
    logger = logging.getLogger()

    return logger


def random_init(seed=0):
    """Set the seed for the random for torch and random package
    Args:
        seed: random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)


def check_para_correctness(cfg):
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    assert os.path.exists(cfg.OUTPUT_DIR), '{} does not exist'.format(cfg.OUTPUT_DIR)

    checkpoints_dir = osp.join(osp.join(cfg.OUTPUT_DIR, 'ckpt'))
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    tfboard_dir = osp.join(osp.join(cfg.OUTPUT_DIR, 'tfboard'))
    if not os.path.exists(tfboard_dir):
        os.makedirs(tfboard_dir)

