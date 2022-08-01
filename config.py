"""Experiment Configuration"""
import os
import re
import glob
import itertools

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('PANet')
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './util']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)
import datetime

@ex.config
def cfg():
    """Default configurations"""
    input_size = (417, 417)
    seed = 1234
    cuda_visable = '0,1,2,3'
    gpu_id = 2
    mode = 'test' # 'train' or 'test'


    if mode == 'train':
        dataset = 'VOC'
        n_steps = 40000
        num_workers = 8
        label_sets = 0
        batch_size = 1
        lr_milestones = [10000, 20000, 30000]
        align_loss_scaler = 1
        base_loss_scaler = 1
        ignore_label = 255
        print_interval = 100
        save_pred_every = 4000
        evaluate_interval = 4000
        n_runs = 1
        eval = 0
        eval_dir='.'
        center = 5
        ckpt_dir = '.'
        skip_ways = 'v1'
        output_sem_size = 417
        infer_max_iters = 1000
        share = 3
        pt_lambda = 0.8
        un_bs = 3
        topk = 30
        global_const = 0.8
        fix = False
        align_loss_cs_scaler = 0
        segments = False
        p_value_thres = 0
        resnet = 101
        output_dir='.'

        model = {
            'part': True,
            'semi': False,
            'sem': True,
            'resnet': True,
            'slic': False,

        }

        task = {
            'n_ways': 1,
            'n_shots': 1,
            'n_queries': 1,
            'n_unlabels': 0,
        }

        optim = {
            'lr': 1e-3,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        }

        slic = {
            'num_components': 80,
            'compactness': 80,
        }


    else:
        raise ValueError('Wrong configuration for "mode" !')

    exp_str = '_'.join(
        [dataset, ]
        + [key for key, value in model.items() if value]
        + [f'w{task["n_ways"]}s{task["n_shots"]}_lr{optim["lr"]}_cen{center}_F{label_sets}'])

    path = {
        'log_dir': './outputs/PANet/',
        'init_path': './FewShotSeg-dataset/cache/vgg16-397923af.pth',
        'VOC':{'data_dir': './FewShotSeg-dataset/Pascal/VOC2012/',
               'data_split': 'trainaug',},
    }

@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook fucntion to add observer"""
    exp_name = f'{ex.path}_{config["exp_str"]}'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config
