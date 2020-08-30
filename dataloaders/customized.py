"""
Customized dataset
"""

import os
import random

import torch
import numpy as np

from .pascal import VOC
from .common import PairedDataset
from fast_slic.avx2 import SlicAvx2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from util.superpixel import *


def attrib_basic(_sample, class_id):
    """
    Add basic attribute

    Args:
        _sample: data sample
        class_id: class label asscociated with the data
            (sometimes indicting from which subset the data are drawn)
    """
    return {'class_id': class_id}


def getMask(label, scribble, class_id, class_ids):
    """
    Generate FG/BG mask from the segmentation mask

    Args:
        label:
            semantic mask
        scribble:
            scribble mask
        class_id:
            semantic class of interest
        class_ids:
            all class id in this episode
    """
    # Dense Mask
    fg_mask = torch.where(label == class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    bg_mask = torch.where(label != class_id,
                          torch.ones_like(label), torch.zeros_like(label)) #set other class to background
    for class_id in class_ids:
        bg_mask[label == class_id] = 0

    # Scribble Mask
    bg_scribble = scribble == 0
    fg_scribble = torch.where((fg_mask == 1)
                              & (scribble != 0)
                              & (scribble != 255),
                              scribble, torch.zeros_like(fg_mask))
    scribble_cls_list = list(set(np.unique(fg_scribble)) - set([0,]))
    if scribble_cls_list:  # Still need investigation
        fg_scribble = fg_scribble == random.choice(scribble_cls_list).item()
    else:
        fg_scribble[:] = 0

    return {'fg_mask': fg_mask,
            'bg_mask': bg_mask,
            'fg_scribble': fg_scribble.long(),
            'bg_scribble': bg_scribble.long()}


def baseOrder(cfg, labels):

    baseclass_ids = np.unique(labels[0])
    baseclass = {}
    label_sets = []
    label_sets.append(list(range(6, 21)))
    label_sets.append(list(range(1, 6)) + list(range(11, 21)))
    label_sets.append(list(range(1, 11)) + list(range(16, 21)))
    label_sets.append(list(range(1, 16)))


    for i in range(len(label_sets)):
        baseclass[i] = {}
        for id_ in range(1, 16):
            baseclass[i][label_sets[i][id_ - 1]] = id_


    ###### Generate query label (class indices in one episode, i.e. the ground truth)######
    query_labels_tmp = [torch.zeros_like(x) for x in labels]

    for i, query_label_tmp in enumerate(query_labels_tmp):
        query_label_tmp[labels[i] == 255] = 255
        for j in range(len(baseclass_ids)):
            if baseclass_ids[j] == 0:
                continue
            elif baseclass_ids[j] == 255 or baseclass_ids[j] not in label_sets[cfg['label_sets']]:
                query_label_tmp[labels[i] == baseclass_ids[j]] = 255
                continue
            query_label_tmp[labels[i] == baseclass_ids[j]] = baseclass[cfg['label_sets']][baseclass_ids[j]]

    return query_labels_tmp


def suppBaseOrder(cfg, labels):
    supp_labels = []
    for way in labels:
        labels_way = []
        for label in way:

            baseclass_ids = np.unique(label)
            baseclass = {}
            label_sets = []
            label_sets.append(list(range(6, 21)))
            label_sets.append(list(range(1, 6)) + list(range(11, 21)))
            label_sets.append(list(range(1, 11)) + list(range(16, 21)))
            label_sets.append(list(range(1, 16)))

            for i in range(len(label_sets)):
                baseclass[i] = {}
                for id_ in range(1, 16):
                    baseclass[i][label_sets[i][id_ - 1]] = id_

            ###### Generate query label (class indices in one episode, i.e. the ground truth)######
            label_tmp = torch.zeros_like(label)

            label_tmp[label == 255] = 255
            for j in range(len(baseclass_ids)):
                if baseclass_ids[j] == 0 or baseclass_ids[j] == 255 or baseclass_ids[j] not in label_sets[cfg['label_sets']]:
                    continue
                elif baseclass_ids[j] == 255 or baseclass_ids[j] not in label_sets[cfg['label_sets']]:
                    label_tmp[label == baseclass_ids[j]] = 255  # 1, 417*417 fewshot label
                    continue
                label_tmp[label == baseclass_ids[j]] = baseclass[cfg['label_sets']][baseclass_ids[j]]  # 1, 417*417 fewshot label

            labels_way.append(label_tmp)

        supp_labels.append(labels_way)


    return supp_labels


def getSBMask(mask, labels):
    """
    Generate FG/BG mask from the segmentation mask

    Args:
        label:
            semantic mask
        scribble:
            scribble mask
        class_id:
            semantic class of interest
        class_ids:
            all class id in this episode
    """
    # Dense Mask
    if 1 not in labels:
        ignore_mask = (mask >= 1) * (mask <= 5)
        mask[mask > 5] -= 5
        mask[ignore_mask==1] = 255
        mask[mask==250] = 255

    elif 6 not in labels:
        ignore_mask = (mask >= 6) * (mask <= 10)
        mask[mask > 10] -= 5
        mask[ignore_mask == 1] = 255
        mask[mask==250] = 255

    elif 11 not in labels:

        ignore_mask = (mask >= 11) * (mask <= 15)
        mask[mask > 15] -= 5
        mask[ignore_mask == 1] = 255
        mask[mask==250] = 255

    elif 16 not in labels:
        ignore_mask = (mask >= 16) * (mask <= 20)
        mask[ignore_mask == 1] = 255

    else:
        x = 1

    return mask

def fewShot(paired_sample, n_ways, n_shots, n_unlabel, cnt_query, coco=False, cfg=None, labels=None):
    """
    Postprocess paired sample for fewshot settings

    Args:
        paired_sample:
            data sample from a PairedDataset
        n_ways:
            n-way few-shot learning
        n_shots:
            n-shot few-shot learning
        cnt_query:
            number of query images for each class in the support set
        coco:
            MS COCO dataset
    """
    ###### Compose the support and query image list ######
    cumsum_idx = np.cumsum([0,] + [n_shots + n_unlabel + x for x in cnt_query])

    # support class ids
    class_ids = [paired_sample[cumsum_idx[i]]['basic_class_id'] for i in range(n_ways)]

    # support images
    support_images = [[paired_sample[cumsum_idx[i] + j]['image'] for j in range(n_shots)] for i in range(n_ways)]
    support_images_t = [[paired_sample[cumsum_idx[i] + j]['image_t'] for j in range(n_shots)] for i in range(n_ways)]

    # support image labels
    if coco:
        support_labels = [[paired_sample[cumsum_idx[i] + j]['label'][class_ids[i]] for j in range(n_shots)] for i in range(n_ways)]
    else:
        support_labels = [[paired_sample[cumsum_idx[i] + j]['label'] for j in range(n_shots)] for i in range(n_ways)] #2, 1, 417*417

    support_scribbles = [[paired_sample[cumsum_idx[i] + j]['scribble'] for j in range(n_shots)] for i in range(n_ways)]
    support_insts = [[paired_sample[cumsum_idx[i] + j]['inst'] for j in range(n_shots)] for i in range(n_ways)]

    # query images, masks and class indices
    query_images = [paired_sample[cumsum_idx[i+1] - j - 1]['image'] for i in range(n_ways) for j in range(cnt_query[i])]
    query_images_t = [paired_sample[cumsum_idx[i+1] - j - 1]['image_t'] for i in range(n_ways) for j in range(cnt_query[i])]
    if coco:
        query_labels = [paired_sample[cumsum_idx[i+1] - j - 1]['label'][class_ids[i]] for i in range(n_ways) for j in range(cnt_query[i])]

    else:
        query_labels = [paired_sample[cumsum_idx[i+1] - j - 1]['label'] for i in range(n_ways) for j in range(cnt_query[i])] #base class
        if cfg['segments']:
            query_segment =[paired_sample[cumsum_idx[i+1]-j-1]['segment'] for i in range(n_ways) for j in range(cnt_query[i])]

    query_cls_idx = [sorted([0,] + [class_ids.index(x) + 1 for x in set(np.unique(query_label)) & set(class_ids)]) for query_label in query_labels]

    ###### Generate support image masks ######
    support_mask = [[getMask(support_labels[way][shot], support_scribbles[way][shot], class_ids[way], class_ids) for shot in range(n_shots)] for way in range(n_ways)]

    '''sb_masks'''
    support_labels_base = suppBaseOrder(cfg, support_labels) #way,shot, 417*417
    query_labels_base = baseOrder(cfg, query_labels) #1, 417*417

    ###### Generate query label (class indices in one episode, i.e. the ground truth)######
    query_labels_tmp = [torch.zeros_like(x) for x in query_labels]
    for i, query_label_tmp in enumerate(query_labels_tmp):
        query_label_tmp[query_labels[i] == 255] = 255
        for j in range(n_ways):
            query_label_tmp[query_labels[i] == class_ids[j]] = j + 1 #1, 417*417 fewshot label[:way] others are 0
    ###### Generate query mask for each semantic class (including BG) ######
    # BG class
    query_masks = [[torch.where(query_label == 0, torch.ones_like(query_label), torch.zeros_like(query_label))[None, ...],] for query_label in query_labels] #1,1,1*417*417 0 and 1
    # Other classes in query image
    for i, query_label in enumerate(query_labels):
        for idx in query_cls_idx[i][1:]:
            mask = torch.where(query_label == class_ids[idx - 1], torch.ones_like(query_label), torch.zeros_like(query_label))[None, ...]
            query_masks[i].append(mask)

    if n_unlabel > 0:
        assert n_unlabel > 0,  "More unlabel images"
        #Unlabel Dataloader
        cumsum_unlabel_idx = cumsum_idx.copy()
        cumsum_unlabel_idx[:n_ways] += n_shots
        unlabel_images = [[paired_sample[cumsum_unlabel_idx[i] + j]['image'] for j in range(n_unlabel)] for i in range(n_ways)]
        unlabel_images_t = [[paired_sample[cumsum_unlabel_idx[i] + j]['image_t'] for j in range(n_unlabel)] for i in range(n_ways)]

        if coco:
            unlabel_labels = [[paired_sample[cumsum_unlabel_idx[i] + j]['label'][class_ids[i]] for j in range(n_unlabel)]for i in range(n_ways)] #2,10,417*417
        else:
            unlabel_labels = [[paired_sample[cumsum_unlabel_idx[i] + j]['label'] for j in range(n_unlabel)] for i in range(n_ways)]  # 2,10,417*417
            if cfg['segments']:
                unlabel_segment = [[paired_sample[cumsum_unlabel_idx[i] + j]['segment'] for j in range(n_unlabel)] for i in range(n_ways)]


        unlabel_labels_tmp = [[torch.zeros_like(y) for y in x] for x in unlabel_labels]
        for i, unlabel_label_tmp in enumerate(unlabel_labels_tmp):
            for k, tmp in enumerate(unlabel_label_tmp):
                tmp[unlabel_labels[i][k] == 255] = 255
                for j in range(n_ways):
                    tmp[unlabel_labels[i][k] == class_ids[j]] = j + 1

    else:

        assert n_unlabel == 0, "the number of unlabel images must be zero"
        ## here we load the pesudo unlabel images to avoid errors
        unlabel_images = query_images
        unlabel_images_t = query_images_t
        unlabel_labels_tmp = query_labels_tmp
        unlabel_spix = query_labels_tmp
        unlabel_segment = query_labels_tmp
        query_segment = query_labels_tmp


    img_name = str(class_ids)

    return {'class_ids': class_ids,

            'support_images_t': support_images_t,
            'support_images': support_images,
            'support_mask': support_mask,
            'support_inst': support_insts,
            'support_labels_base': support_labels_base,

            'query_images_t': query_images_t,
            'query_images': query_images,
            'query_labels': query_labels_tmp,
            'query_masks': query_masks,
            'query_cls_idx': query_cls_idx,
            'query_labels_base': query_labels_base,
            'query_segment': query_segment,

            'img_name': img_name,

            'unlabel_images_t': unlabel_images_t,
            'unlabel_images': unlabel_images,
            'unlabel_labels': unlabel_labels_tmp,
            'unlabel_segment': unlabel_segment,

            'cnt_query': cnt_query

            }


def voc_fewshot(base_dir, split, transforms, to_tensor, labels, n_ways, n_shots, max_iters,
                n_queries=1, n_unlabel=0, cfg=None):
    """
    Args:
        base_dir:
            VOC dataset directory
        split:
            which split to use
            choose from ('train', 'val', 'trainval', 'trainaug')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
        labels:
            object class labels of the data
        n_ways:
            n-way few-shot learning, should be no more than # of object class labels
        n_shots:
            n-shot few-shot learning
        max_iters:
            number of pairs
        n_queries:
            number of query images
    """
    voc = VOC(base_dir=base_dir, split=split, transforms=transforms, to_tensor=to_tensor)
    voc.add_attrib('basic', attrib_basic, {})

    # Load image ids for each class
    sub_ids = []
    for label in labels:
        with open(os.path.join(voc._id_dir, voc.split,
                               'class{}.txt'.format(label)), 'r') as f:
            sub_ids.append(f.read().splitlines())
    # Create sub-datasets and add class_id attribute
    subsets = voc.subsets(sub_ids, [{'basic': {'class_id': cls_id}} for cls_id in labels])

    # Choose the classes of queries
    cnt_query = np.bincount(random.choices(population=range(n_ways), k=n_queries), minlength=n_ways)
    # Set the number of images for each class
    n_elements = [n_shots + n_unlabel + x for x in cnt_query]
    # Create paired dataset
    paired_data = PairedDataset(subsets, n_elements=n_elements, max_iters=max_iters, same=False,
                                pair_based_transforms=[
                                    (fewShot, {'n_ways': n_ways, 'n_shots': n_shots, 'n_unlabel': n_unlabel,
                                               'cnt_query': cnt_query, 'cfg' : cfg, 'labels': labels})])
    return paired_data

