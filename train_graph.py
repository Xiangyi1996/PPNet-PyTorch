# -*- coding: utf-8 -*-
# @Author  : Xiangyi Zhang
# @File    : train_graph.py
# @Email   : zhangxy9@shanghaitech.edu.cn
import os
import shutil

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose
from models.SemiFewShotPartGraph import SemiFewShotSegPartGraph

from dataloaders.customized import voc_fewshot
from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS, get_params
from config import ex
from util.metric import Metric
import tqdm
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from util.utils import check_dir
import numpy as np

import time
import os.path as osp
import pprint

@ex.automain
def main(_run, _config, _log):

    logdir = f'{_run.observers[0].dir}/'
    print(logdir)
    category = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
    training_tags = {'loss': "ATraining/total_loss", "query_loss": "ATraining/query_loss",
                     'aligned_loss': "ATraining/aligned_loss", 'base_loss': "ATraining/base_loss",}
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')
    cfg = _config
    data_name = _config['dataset']
    max_label=20 if data_name == 'VOC' else 80

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'###### Setting: {_run.observers[0].dir} ######')
    print(_config['ckpt_dir'])
    tbwriter = SummaryWriter(osp.join(_config['ckpt_dir']))
    infer_tags = {
        'mean_iou': "MeanIoU/mean_iou",
        "mean_iou_binary": "MeanIoU/mean_iou_binary",
    }

    _log.info('###### Create model ######')
    model = SemiFewShotSegPartGraph(cfg=_config)
    _log.info('Model: SemiFewShotSegPartGraph')

    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
    model.train()

    _log.info('###### Load data ######')
    data_name = _config['dataset']
    make_data = voc_fewshot
    labels = CLASS_LABELS[data_name][_config['label_sets']]
    transforms = Compose([Resize(size=_config['input_size']),RandomMirror()])
    dataset = make_data(
        base_dir=_config['path'][data_name]['data_dir'],
        split=_config['path'][data_name]['data_split'],
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,
        max_iters=_config['n_steps'] * _config['batch_size'],
        n_ways=_config['task']['n_ways'],
        n_shots=_config['task']['n_shots'],
        n_queries=_config['task']['n_queries'],
        n_unlabel=_config['task']['n_unlabels'],
        cfg=_config

    )
    trainloader = DataLoader(
        dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    # Optimizer
    if _config['fix']:
        print('Optimizer: fix')
        optimizer = torch.optim.SGD(
            params=[
                {
                    "params": model.module.encoder.layer3.parameters(),
                    "lr": _config['optim']['lr'],
                    "weight_decay": _config['optim']['weight_decay']
                },
                {
                    "params": model.module.encoder.layer4.parameters(),
                    "lr": _config['optim']['lr'],
                    "weight_decay": _config['optim']['weight_decay']
                },
                {
                    "params": get_params(model, key='20x'),
                    "lr": 20 * _config['optim']['lr'],
                    "weight_decay": 0
                },
                {
                    "params": get_params(model, key='10x'),
                    "lr": 10 * _config['optim']['lr'],
                    "weight_decay": _config['optim']['weight_decay']
                }], lr=_config['optim']['lr'], weight_decay=_config['optim']['weight_decay'],
            momentum=_config['optim']['momentum'])

    else:
        print('Optimizer: Not fix')
        optimizer = torch.optim.SGD(
            params=[
                {
                    "params": model.module.encoder.parameters(),
                    "lr": _config['optim']['lr'],
                    "weight_decay": _config['optim']['weight_decay']
                },
                {
                    "params": get_params(model, key='20x'),
                    "lr": 20 * _config['optim']['lr'],
                    "weight_decay": 0
                },
                {
                    "params": get_params(model, key='10x'),
                    "lr": 10 * _config['optim']['lr'],
                    "weight_decay": _config['optim']['weight_decay']
                }], lr=_config['optim']['lr'], weight_decay=_config['optim']['weight_decay'], momentum=_config['optim']['momentum'])

    print('scheduler: MultiStepLR')
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)

    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])

    log_loss = {'loss': 0, 'align_loss': 0, 'base_loss': 0, 'align_loss_cs': 0}
    _log.info('###### Training ######')

    highest_iou = 0
    metrics = {}
    device = torch.device('cuda')

    for i_iter, sample_batched in enumerate(trainloader):
        if _config['fix']:
            model.module.encoder.conv1.eval()
            model.module.encoder.bn1.eval()
            model.module.encoder.layer1.eval()
            model.module.encoder.layer2.eval()

        if _config['eval']:
            if i_iter == 0:
                break
        # Prepare input
        support_images = [[shot.cuda() for shot in way] for way in sample_batched['support_images']]
        support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way] for way in sample_batched['support_mask']]
        support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way] for way in sample_batched['support_mask']]

        query_images = [query_image.cuda() for query_image in sample_batched['query_images']]
        query_labels = torch.cat([query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)#1*417*417

        unlabel_images = [[unlabel_images.cuda() for unlabel_images in way] for way in sample_batched['unlabel_images']]
        unlabel_segment = [[un.float().to(device) for un in way] for way in sample_batched['unlabel_segment']]

        # Forward and Backward
        optimizer.zero_grad()
        query_pred, output_semantic, align_loss = model(support_images, support_fg_mask, support_bg_mask, query_images, unlabel_images, unlabel_segment)

        support_label_base = torch.cat([torch.cat([shot.long().cuda() for shot in way]) for way in sample_batched['support_labels_base']])  # 2*1*417*417
        query_labels_base = torch.cat([query_label.long().cuda() for query_label in sample_batched['query_labels_base']], dim=0)  # 1*417*417
        label_base = torch.cat((support_label_base, query_labels_base))  # 3*53*53

        query_loss = criterion(query_pred, query_labels) #1*3*417*417, 1*417*417
        if cfg['model']['sem']:
            base_loss = criterion(output_semantic, label_base)  # 3*16*53*53, 3*53*53
        else:
            base_loss = torch.zeros(1).to(device)

        loss = query_loss + align_loss * _config['align_loss_scaler'] + base_loss * _config['base_loss_scaler']
        loss.backward()
        optimizer.step()
        scheduler.step(epoch=i_iter)

        # Log loss
        query_loss = query_loss.detach().data.cpu().numpy()
        align_loss = align_loss.detach().data.cpu().numpy()
        base_loss = base_loss.detach().data.cpu().numpy()

        # _run.log_scalar('loss', query_loss)
        # _run.log_scalar('align_loss', align_loss)
        log_loss['loss'] += query_loss
        log_loss['align_loss'] += align_loss
        log_loss['base_loss'] += base_loss

        # print loss and take snapshots
        if (i_iter + 1) % _config['print_interval'] == 0:
            loss = log_loss['loss'] / (i_iter + 1)
            align_loss = log_loss['align_loss'] / (i_iter + 1)
            base_loss = log_loss['base_loss'] / (i_iter + 1)

            print(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss}, base_loss: {base_loss}')
            _log.info(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss}, base_loss: {base_loss}')

            metrics['loss'] = loss
            metrics['query_loss'] = query_loss
            metrics['align_loss'] = align_loss
            metrics['base_loss'] = base_loss

            # for k, v in metrics.items():
            #     tbwriter.add_scalar(training_tags[k], v, i_iter)

        if (i_iter + 1) % _config['evaluate_interval'] == 0:
            _log.info('###### Evaluation begins ######')
            _log.info(f'###### Setting: {_run.observers[0].dir} ######')
            print(_config['ckpt_dir'])

            model.eval()

            labels = CLASS_LABELS[data_name]['all'] - CLASS_LABELS[data_name][_config['label_sets']]
            transforms = [Resize(size=_config['input_size'])]
            transforms = Compose(transforms)

            metric = Metric(max_label=max_label, n_runs=_config['n_runs'])
            with torch.no_grad():
                for run in range(1):
                    _log.info(f'### Run {run + 1} ###')
                    set_seed(_config['seed'] + run)

                    _log.info(f'### Load data ###')
                    dataset = make_data(
                        base_dir=_config['path'][data_name]['data_dir'],
                        split=_config['path'][data_name]['data_split'],
                        transforms=transforms,
                        to_tensor=ToTensorNormalize(),
                        labels=labels,
                        max_iters=_config['infer_max_iters'],
                        n_ways=_config['task']['n_ways'],
                        n_shots=_config['task']['n_shots'],
                        n_queries=_config['task']['n_queries'],
                        n_unlabel=_config['task']['n_unlabels'],
                        cfg=_config

                    )
                    testloader = DataLoader(dataset, batch_size=_config['batch_size'], shuffle=False,
                                            num_workers=_config['num_workers'], pin_memory=True, drop_last=False)
                    _log.info(f"Total # of Data: {len(dataset)}")

                    for sample_batched in tqdm.tqdm(testloader):
                        label_ids = list(sample_batched['class_ids'])
                        support_images = [[shot.cuda() for shot in way]
                                          for way in sample_batched['support_images']]
                        suffix = 'mask'

                        support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                                           for way in sample_batched['support_mask']]
                        support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                                           for way in sample_batched['support_mask']]

                        unlabel_images = [[unlabel_images.cuda() for unlabel_images in way] for way in
                                          sample_batched['unlabel_images']]
                        unlabel_segment = [[un.float().to(device) for un in way] for way in
                                           sample_batched['unlabel_segment']]

                        query_images = [query_image.cuda()
                                        for query_image in sample_batched['query_images']]
                        query_labels = torch.cat([query_label.cuda() for query_label in sample_batched['query_labels']], dim=0)

                        query_pred, _, _ = model(support_images, support_fg_mask, support_bg_mask, query_images, unlabel_images, unlabel_segment)

                        curr_iou = metric.record(query_pred.argmax(dim=1)[0], query_labels[0], labels=label_ids, n_run=run)

                    classIoU, meanIoU = metric.get_mIoU(labels=sorted(labels), n_run=run)
                    classIoU_binary, meanIoU_binary = metric.get_mIoU_binary(n_run=run)

                    _run.log_scalar('classIoU', classIoU.tolist())
                    _run.log_scalar('meanIoU', meanIoU.tolist())
                    _run.log_scalar('classIoU_binary', classIoU_binary.tolist())
                    _run.log_scalar('meanIoU_binary', meanIoU_binary.tolist())
                    _log.info(f'classIoU: {classIoU}')
                    _log.info(f'meanIoU: {meanIoU}')
                    _log.info(f'classIoU_binary: {classIoU_binary}')
                    _log.info(f'meanIoU_binary: {meanIoU_binary}')

                    print(f'meanIoU: {meanIoU}, meanIoU_binary: {meanIoU_binary}')

                    metrics = {}

                    metrics['mean_iou'] = meanIoU
                    metrics['mean_iou_binary'] = meanIoU_binary

                    for k, v in metrics.items():
                        tbwriter.add_scalar(infer_tags[k], v, i_iter)

                    if meanIoU > highest_iou:
                        print(f'The highest iou is in iter: {i_iter} : {meanIoU}, save: {_config["ckpt_dir"]}/best.pth')
                        highest_iou = meanIoU
                        torch.save(model.state_dict(), os.path.join(f'{_config["ckpt_dir"]}/best.pth'))
                    else:
                        print(f'The highest iou is in iter: {i_iter} : {meanIoU}')
            torch.save(model.state_dict(), os.path.join(f'{_config["ckpt_dir"]}/{i_iter + 1}.pth'))

        model.train()

    _log.info(f'###### Setting: {_run.observers[0].dir} ######')
    print(_config['ckpt_dir'])


    _log.info(' --------- Testing begins ---------')

    labels = CLASS_LABELS[data_name]['all'] - CLASS_LABELS[data_name][_config['label_sets']]
    transforms = [Resize(size=_config['input_size'])]
    transforms = Compose(transforms)
    ckpt = os.path.join(f'{_config["ckpt_dir"]}/best.pth')

    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()

    metric = Metric(max_label=max_label, n_runs=5)
    with torch.no_grad():
        for run in range(5):
            n_iter = 0
            _log.info(f'### Run {run + 1} ###')
            set_seed(_config['seed'] + run)

            _log.info(f'### Load data ###')
            dataset = make_data(
                base_dir=_config['path'][data_name]['data_dir'],
                split=_config['path'][data_name]['data_split'],
                transforms=transforms,
                to_tensor=ToTensorNormalize(),
                labels=labels,
                max_iters=_config['infer_max_iters'],
                n_ways=_config['task']['n_ways'],
                n_shots=_config['task']['n_shots'],
                n_queries=_config['task']['n_queries'],
                n_unlabel=_config['task']['n_unlabels'],
                cfg=_config

            )
            testloader = DataLoader(dataset, batch_size=_config['batch_size'], shuffle=False,
                                    num_workers=_config['num_workers'], pin_memory=True, drop_last=False)
            _log.info(f"Total # of Data: {len(dataset)}")

            for sample_batched in tqdm.tqdm(testloader):

                label_ids = list(sample_batched['class_ids'])
                support_images = [[shot.cuda() for shot in way]
                                  for way in sample_batched['support_images']]
                suffix = 'mask'

                support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                                   for way in sample_batched['support_mask']]
                support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                                   for way in sample_batched['support_mask']]

                unlabel_images = [[unlabel_images.cuda() for unlabel_images in way] for way in
                                  sample_batched['unlabel_images']]
                unlabel_segment = [[un.float().to(device) for un in way] for way in
                                   sample_batched['unlabel_segment']]

                query_images = [query_image.cuda()
                                for query_image in sample_batched['query_images']]
                query_labels = torch.cat([query_label.cuda() for query_label in sample_batched['query_labels']], dim=0)

                query_pred, _, _ = model(support_images, support_fg_mask, support_bg_mask, query_images, unlabel_images, unlabel_segment)

                curr_iou = metric.record(query_pred.argmax(dim=1)[0], query_labels[0], labels=label_ids, n_run=run)

                n_iter += 1

            classIoU, meanIoU = metric.get_mIoU(labels=sorted(labels), n_run=run)
            classIoU_binary, meanIoU_binary = metric.get_mIoU_binary(n_run=run)

            _run.log_scalar('classIoU', classIoU.tolist())
            _run.log_scalar('meanIoU', meanIoU.tolist())
            _run.log_scalar('classIoU_binary', classIoU_binary.tolist())
            _run.log_scalar('meanIoU_binary', meanIoU_binary.tolist())
            _log.info(f'classIoU: {classIoU}')
            _log.info(f'meanIoU: {meanIoU}')
            _log.info(f'classIoU_binary: {classIoU_binary}')
            _log.info(f'meanIoU_binary: {meanIoU_binary}')

    classIoU, classIoU_std, meanIoU, meanIoU_std = metric.get_mIoU(labels=sorted(labels))
    classIoU_binary, classIoU_std_binary, meanIoU_binary, meanIoU_std_binary = metric.get_mIoU_binary()

    _run.log_scalar('meanIoU', meanIoU.tolist())
    _run.log_scalar('meanIoU_binary', meanIoU_binary.tolist())

    _run.log_scalar('final_classIoU', classIoU.tolist())
    _run.log_scalar('final_classIoU_std', classIoU_std.tolist())
    _run.log_scalar('final_meanIoU', meanIoU.tolist())
    _run.log_scalar('final_meanIoU_std', meanIoU_std.tolist())
    _run.log_scalar('final_classIoU_binary', classIoU_binary.tolist())
    _run.log_scalar('final_classIoU_std_binary', classIoU_std_binary.tolist())
    _run.log_scalar('final_meanIoU_binary', meanIoU_binary.tolist())
    _run.log_scalar('final_meanIoU_std_binary', meanIoU_std_binary.tolist())


    _log.info('----- Final Result -----')
    _log.info(f'classIoU mean: {classIoU}')
    _log.info(f'classIoU std: {classIoU_std}')
    _log.info(f'meanIoU mean: {meanIoU}')
    _log.info(f'meanIoU std: {meanIoU_std}')
    _log.info(f'classIoU_binary mean: {classIoU_binary}')
    _log.info(f'classIoU_binary std: {classIoU_std_binary}')
    _log.info(f'meanIoU_binary mean: {meanIoU_binary}')
    _log.info(f'meanIoU_binary std: {meanIoU_std_binary}')

    _log.info("## ------------------------------------------ ##")
    _log.info(f'###### Setting: {_run.observers[0].dir} ######')

    _log.info("Running {num_run} runs, meanIoU:{miou:.4f}, meanIoU_binary:{mbiou:.4f} "
                     "meanIoU_std:{miou_std:.4f}, meanIoU_binary_std:{mbiou_std:.4f}".format(
        num_run=5, miou=meanIoU, mbiou=meanIoU_binary, miou_std=meanIoU_std,
        mbiou_std=meanIoU_std_binary))
    _log.info(f"Current setting is {_run.observers[0].dir}")

    print("Running {num_run} runs, meanIoU:{miou:.4f}, meanIoU_binary:{mbiou:.4f} "
                     "meanIoU_std:{miou_std:.4f}, meanIoU_binary_std:{mbiou_std:.4f}".format(
        num_run=5, miou=meanIoU, mbiou=meanIoU_binary, miou_std=meanIoU_std,
        mbiou_std=meanIoU_std_binary))
    print(f"Current setting is {_run.observers[0].dir}")
    print(_config['ckpt_dir'])
    print(logdir)


